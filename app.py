import os
import json
import re
import asyncio
import httpx
import tempfile
import PyPDF2
from PIL import Image
import pytesseract
import docx
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from typing import Dict

# --- Set Tesseract path for local development ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()

# Mount static files directory
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# URL for your local Ollama instance's API
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Load the Packaging Knowledge Graph
try:
    with open("pack_graph.json", "r") as f:
        PACK_GRAPH = json.load(f)
except FileNotFoundError:
    PACK_GRAPH = {}

# In-memory storage for feedback, simulating a database
feedback_log = []

# New in-memory bin database
bin_data = {
    "bin-A": {"capacity_kg": 25.0, "fill_level_kg": 0.0, "location": (40.71, -74.00)},  # Manhattan
    "bin-B": {"capacity_kg": 25.0, "fill_level_kg": 0.0, "location": (34.05, -118.24)}, # Los Angeles
    "bin-C": {"capacity_kg": 50.0, "fill_level_kg": 0.0, "location": (41.87, -87.62)}, # Chicago
    "bin-D": {"capacity_kg": 50.0, "fill_level_kg": 0.0, "location": (29.76, -95.36)}  # Houston
}

# --- NEW: In-memory user credit database and its persistence ---
user_credits: Dict[str, float] = {}
CREDIT_RATE = 1.0 # 1 credit per kg of plastic

# Load existing user credits from a file on startup
def load_credits_db():
    global user_credits
    if os.path.exists("credits_db.json"):
        with open("credits_db.json", "r") as f:
            user_credits = json.load(f)

# Save user credits to a file
def save_credits_db():
    with open("credits_db.json", "w") as f:
        json.dump(user_credits, f, indent=2)

load_credits_db()

# --- Pydantic models for request bodies ---
class ProcessedData(BaseModel):
    classified_items: list
    bag_recipes: list

class ManifestFeedback(BaseModel):
    manifest_id: str
    collector_status: str
    timestamp: str

class BinFeedback(BaseModel):
    bin_id: str
    collector_status: str
    timestamp: str

class CreditDeposit(BaseModel):
    user_id: str
    waste_type: str
    weight_kg: float
    timestamp: str

class CreditUser(BaseModel):
    user_id: str
    balance: float

# Helper Functions (unchanged, for brevity)
def extract_text(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        return pytesseract.image_to_string(Image.open(file_path))
    elif ext == ".pdf":
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext == ".txt":
        return open(file_path, "r", encoding="utf-8").read()
    elif ext == ".csv":
        return ""
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data)
    else:
        return ""

def clean_items(raw_text):
    items = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(word in line.lower() for word in ["total", "price", "amount", "subtotal", "tax"]):
            continue
        if re.match(r'^\$?\d+(\.\d+)?$', line):
            continue
        line = re.sub(r'\$?\d+(\.\d+)?(/\w+)?', "", line)
        line = re.sub(r'\b\d+\s*(lbs?|kg|g|dozen|box|pack|bag|cups?|loaves?|gallon|pk)\b', "", line, flags=re.IGNORECASE)
        line = re.sub(r'[^a-zA-Z\s]', " ", line)
        line = re.sub(r'\s+', " ", line).strip()
        if len(line) < 2:
            continue
        items.append(line)
    return list(dict.fromkeys(items))

def generate_bag_recipe(classified_items):
    streams = {}
    for item in classified_items:
        stream = item.get("stream", "Unknown")
        streams.setdefault(stream, []).append(item)
    bag_recipes = []
    for stream, items in streams.items():
        bag_count = max(1, (len(items) + 9) // 10)
        instructions = []
        for itm in items:
            item_name = itm.get("item", "Unknown Item").strip()
            if not item_name:
                item_name = "Unknown Item"
            note = itm.get("note", "Check item before disposal.")
            instructions.append({"item": item_name, "note": note})
        bag_recipes.append({
            "stream": stream,
            "bag_count": bag_count,
            "instructions": instructions
        })
    return bag_recipes

def generate_manifest(classified_items, bag_recipes, bin_id):
    manifest_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    total_weight = sum(item.get('weight_kg', 0) for item in classified_items)
    return {
        "manifest_id": manifest_id,
        "timestamp": timestamp,
        "location": bin_data.get(bin_id, {}).get("location"),
        "total_items": len(classified_items),
        "total_bags": sum(bag['bag_count'] for bag in bag_recipes),
        "total_weight_kg": round(total_weight, 2),
        "bag_recipes": bag_recipes,
        "classified_items": classified_items
    }

async def classify_with_llm(item, client: httpx.AsyncClient):
    prompt = f"""You are a waste packaging classifier. Analyze the item and respond with ONLY valid JSON, no markdown, no code blocks, no explanations.

Required JSON format:
{{
  "category": "PET|Glass|Paper|Metal|MLP|Compost|Other",
  "stream": "Dry|Wet|Recyclable|None",
  "recyclability": "High|Moderate|Low|None",
  "weight_kg": 0.01
}}

Item to classify: "{item}"

Respond with ONLY the JSON object, nothing else."""
    
    # Try multiple models in order of preference
    models = ["llama3.2", "llama3", "mistral", "deepseekv3.1:671b-cloud"]
    
    for model in models:
        try:
            response = await client.post(
                OLLAMA_API_URL, 
                json={"model": model, "prompt": prompt, "stream": False}, 
                timeout=30.0
            )
            response.raise_for_status()
            raw_output = response.json().get("response", "").strip()
            
            # Remove markdown code blocks if present
            if raw_output.startswith("```"):
                # Extract JSON from code blocks
                lines = raw_output.split("\n")
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if not in_code_block or (in_code_block and line.strip()):
                        json_lines.append(line)
                raw_output = "\n".join(json_lines).strip()
            
            # Try to find JSON object in the response
            json_start = raw_output.find("{")
            json_end = raw_output.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                raw_output = raw_output[json_start:json_end]
            
            # Parse JSON
            parsed = json.loads(raw_output)
            
            # Validate required fields
            if all(key in parsed for key in ["category", "stream", "recyclability", "weight_kg"]):
                return {"item": item, **parsed}
            else:
                continue  # Try next model
                
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
            # Try next model
            continue
        except Exception as e:
            # Try next model
            continue
    
    # If all models failed, return error
    return {"item": item, "error": "All LLM classification attempts failed"}

def estimate_weight_from_name(item_name):
    """Estimate weight based on item name, quantities, and units"""
    item_lower = item_name.lower().strip()
    base_weight = 0.1  # Default base weight in kg
    
    # Extract quantities and units
    import re
    
    # Check for weight units (lbs, kg, g, oz)
    weight_match = re.search(r'(\d+(?:\.\d+)?)\s*(lbs?|pounds?|kg|kilograms?|g|grams?|oz|ounces?)', item_lower)
    if weight_match:
        value = float(weight_match.group(1))
        unit = weight_match.group(2).lower()
        if unit in ['lb', 'lbs', 'pound', 'pounds']:
            return value * 0.453592  # Convert lbs to kg
        elif unit in ['kg', 'kilogram', 'kilograms']:
            return value
        elif unit in ['g', 'gram', 'grams']:
            return value / 1000.0
        elif unit in ['oz', 'ounce', 'ounces']:
            return value * 0.0283495
    
    # Check for volume units (gallon, liter, ml, fl oz)
    volume_match = re.search(r'(\d+(?:\.\d+)?)\s*(gallon|gal|liter|litre|l|ml|milliliter|fl\s*oz|fluid\s*ounce)', item_lower)
    if volume_match:
        value = float(volume_match.group(1))
        unit = volume_match.group(2).lower()
        if 'gallon' in unit or 'gal' in unit:
            # Estimate: 1 gallon of liquid ≈ 3.8 kg
            return value * 3.8
        elif 'liter' in unit or unit == 'l':
            return value * 1.0  # 1 liter ≈ 1 kg for most liquids
        elif 'ml' in unit or 'milliliter' in unit:
            return value * 0.001
        elif 'fl oz' in unit or 'fluid ounce' in unit:
            return value * 0.0295735
    
    # Check for count-based items
    count_match = re.search(r'(\d+)\s*(dozen|pack|packs?|box|boxes|bag|bags|loaf|loaves|bunch|bunches)', item_lower)
    if count_match:
        count = int(count_match.group(1))
        unit = count_match.group(2).lower()
        
        if 'dozen' in unit:
            # 1 dozen eggs ≈ 0.7 kg, adjust for other items
            if 'egg' in item_lower:
                return count * 0.7
            else:
                return count * 0.5
        elif 'loaf' in unit or 'loaves' in unit:
            return count * 0.5  # 1 loaf of bread ≈ 0.5 kg
        elif 'box' in unit or 'boxes' in unit:
            return count * 0.3  # Average box weight
        elif 'bag' in unit or 'bags' in unit:
            return count * 0.2  # Average bag weight
        elif 'pack' in unit or 'packs' in unit:
            return count * 0.15
        elif 'bunch' in unit or 'bunches' in unit:
            return count * 0.3
    
    # Category-based weight estimation
    if any(kw in item_lower for kw in ['chicken', 'meat', 'beef', 'pork', 'turkey']):
        # Meat is heavy - estimate 0.5-1 kg per item if no quantity specified
        return 0.7
    elif 'gallon' in item_lower or 'milk' in item_lower:
        return 3.8  # 1 gallon of milk
    elif 'dozen' in item_lower and 'egg' in item_lower:
        return 0.7
    elif 'loaf' in item_lower or 'loaves' in item_lower:
        return 0.5
    elif any(kw in item_lower for kw in ['apple', 'banana', 'orange', 'fruit']):
        # Individual fruits: 0.15-0.2 kg each, but if plural assume multiple
        if any(kw in item_lower for kw in ['apples', 'bananas', 'oranges']):
            return 1.0  # Multiple fruits
        return 0.15
    elif any(kw in item_lower for kw in ['tomato', 'tomatoes', 'potato', 'potatoes']):
        if any(kw in item_lower for kw in ['tomatoes', 'potatoes']):
            return 0.8  # Multiple
        return 0.2
    elif 'box' in item_lower:
        return 0.3
    elif 'bag' in item_lower:
        return 0.2
    elif 'cup' in item_lower or 'cups' in item_lower:
        return 0.1
    elif 'bottle' in item_lower:
        return 0.5  # Average bottle
    
    return base_weight

def classify_with_fallback(item):
    """Fallback classification based on keywords when LLM fails"""
    item_lower = item.lower().strip()
    
    # Estimate weight first
    estimated_weight = estimate_weight_from_name(item)
    
    # Food items -> Compost
    food_keywords = ["apple", "banana", "bread", "egg", "chicken", "meat", "fish", "vegetable", "fruit", "tomato", "salad", "pasta", "rice"]
    if any(kw in item_lower for kw in food_keywords):
        return {
            "item": item,
            "category": "Compost",
            "stream": "Wet",
            "recyclability": "High",
            "note": "Dispose in compost bin.",
            "weight_kg": round(estimated_weight, 3)
        }
    
    # Paper items
    paper_keywords = ["box", "paper", "cardboard", "newspaper", "magazine"]
    if any(kw in item_lower for kw in paper_keywords):
        return {
            "item": item,
            "category": "Paper",
            "stream": "Dry",
            "recyclability": "High",
            "note": "Flatten before disposal.",
            "weight_kg": round(estimated_weight, 3)
        }
    
    # Plastic items
    plastic_keywords = ["bottle", "cup", "container", "bag", "wrap", "packaging"]
    if any(kw in item_lower for kw in plastic_keywords):
        return {
            "item": item,
            "category": "PET",
            "stream": "Recyclable",
            "recyclability": "High",
            "note": "Rinse before disposal.",
            "weight_kg": round(estimated_weight, 3)
        }
    
    # Default fallback
    return {
        "item": item,
        "category": "Other",
        "stream": "None",
        "recyclability": "Low",
        "note": "Check item before disposal.",
        "weight_kg": round(estimated_weight, 3)
    }

async def classify_item(item, client):
    item_lower = item.lower().strip()
    
    # First check knowledge graph
    for key, data in PACK_GRAPH.items():
        if key in item_lower:
            return {
                "item": item,
                "category": data["category"],
                "stream": data["stream"],
                "recyclability": data["recyclability"],
                "note": data["note"],
                "weight_kg": data.get("weight_kg", 0.01)
            }

    # Try LLM classification
    llm_result = await classify_with_llm(item, client)
    
    # If LLM failed, use fallback
    if "error" in llm_result:
        return classify_with_fallback(item)
    
    # Validate and clean LLM result
    stream = llm_result.get("stream", "Unknown")
    category = llm_result.get("category", "Other")
    
    # Generate appropriate note
    note = "Check item before disposal."
    if stream == "Wet":
        note = "Dispose as wet compost."
    elif stream == "Dry":
        note = "Dispose as dry recyclables."
    elif stream == "Recyclable":
        if category in ["Glass", "Metal"]:
            note = "Rinse & remove lid if applicable."
        else:
            note = "Rinse & flatten."
    
    weight = llm_result.get("weight_kg")
    if weight is None or not isinstance(weight, (int, float)) or weight <= 0:
        # Use smart weight estimation if LLM didn't provide valid weight
        weight = estimate_weight_from_name(item)
    else:
        weight = float(weight)
    
    return {
        "item": item,
        "category": category,
        "stream": stream,
        "recyclability": llm_result.get("recyclability", "Moderate"),
        "note": note,
        "weight_kg": round(weight, 3)
    }

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def solve_tsp(locations):
    if len(locations) < 2:
        return {"path": locations, "distance": 0.0}

    depot = (40.71, -74.00)
    all_locations = [depot] + locations

    manager = pywrapcp.RoutingIndexManager(len(all_locations), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(calculate_distance(all_locations[from_node], all_locations[to_node]) * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(all_locations[node_index])
            index = solution.Value(routing.NextVar(index))
        node_index = manager.IndexToNode(index)
        route.append(all_locations[node_index])
        
        total_distance = solution.ObjectiveValue() / 1000
        return {"path": route, "distance": round(total_distance, 2)}
    else:
        return {"path": locations, "distance": -1}

def save_classified_data(data):
    try:
        if os.path.exists("classification_db.json"):
            with open("classification_db.json", "r") as f:
                db = json.load(f)
        else:
            db = []
        db.append(data)
        with open("classification_db.json", "w") as f:
            json.dump(db, f, indent=2)
    except Exception as e:
        print(f"Error saving classified data: {e}")

# ---------- API Endpoints ----------
@app.post("/process_file")
async def process_file(file: UploadFile = File(...), bin_id: str = Form(...)):
    if bin_id not in bin_data:
        return JSONResponse({"error": "Invalid bin_id provided."}, status_code=400)

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name
    try:
        raw_text = extract_text(temp_path)
        if not raw_text.strip():
            return JSONResponse({"error": "No text found in file"}, status_code=400)
        items = clean_items(raw_text)
        if not items:
            return JSONResponse({"error": "No valid items found after cleaning"}, status_code=400)

        async with httpx.AsyncClient() as client:
            tasks = [classify_item(item, client) for item in items]
            results = await asyncio.gather(*tasks, return_exceptions=False)
        
        bag_recipes = generate_bag_recipe(results)
        manifest = generate_manifest(results, bag_recipes, bin_id)
        
        save_classified_data({
            "timestamp": datetime.now().isoformat(),
            "bin_id": bin_id,
            "items": results
        })

        bin_data[bin_id]["fill_level_kg"] += manifest["total_weight_kg"]

        return JSONResponse({"classified_items": results, "bag_recipes": bag_recipes, "manifest": manifest})
    finally:
        os.remove(temp_path)

@app.post("/feedback")
async def receive_feedback(feedback: ManifestFeedback):
    feedback_log.append(feedback.dict())
    print(f"Received feedback for Manifest ID {feedback.manifest_id}: Status is {feedback.collector_status}")
    return {"message": "Feedback received successfully", "manifest_id": feedback.manifest_id}

@app.post("/bin_feedback")
async def receive_bin_feedback(feedback: BinFeedback):
    feedback_log.append(feedback.dict())
    if feedback.collector_status == "Valid" and feedback.bin_id in bin_data:
        bin_data[feedback.bin_id]["fill_level_kg"] = 0.0
    print(f"Received feedback for Bin ID {feedback.bin_id}: Status is {feedback.collector_status}")
    return {"message": "Bin feedback received successfully", "bin_id": feedback.bin_id}

@app.get("/analytics")
async def get_analytics():
    return JSONResponse({"feedback_data": feedback_log, "bin_status": bin_data})

@app.get("/optimize_routes")
async def optimize_routes():
    pickup_locations = []
    for bin_id, data in bin_data.items():
        fill_percentage = (data["fill_level_kg"] / data["capacity_kg"]) * 100
        if fill_percentage >= 75:
            pickup_locations.append(data["location"])
    
    if not pickup_locations:
        return JSONResponse({"message": "No bins are ready for pickup."})
    
    route_result = solve_tsp(pickup_locations)
    
    return JSONResponse({"message": "Pickup route optimized.", "route": route_result})

# --- NEW: Credit System Endpoints ---
@app.post("/deposit_recyclable")
async def deposit_recyclable_plastic(deposit: CreditDeposit):
    if deposit.waste_type.lower() != "recyclable plastics":
        return JSONResponse({"message": "Incorrect waste type. Only 'Recyclable Plastics' are accepted for credits."}, status_code=400)
    
    credits_earned = deposit.weight_kg * CREDIT_RATE
    user_credits[deposit.user_id] = user_credits.get(deposit.user_id, 0.0) + credits_earned
    save_credits_db() # Persist the changes

    return JSONResponse({
        "message": "Deposit successful",
        "user_id": deposit.user_id,
        "credits_earned": credits_earned,
        "new_balance": user_credits[deposit.user_id]
    })

@app.get("/user_balance/{user_id}")
async def get_user_balance(user_id: str):
    balance = user_credits.get(user_id, 0.0)
    return JSONResponse({"user_id": user_id, "balance": balance})

# HTML Routes
@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>WasteWise Dashboard</h1><p>Please ensure static/index.html exists</p>", status_code=404)

@app.get("/collector", response_class=HTMLResponse)
async def collector_page():
    try:
        with open("static/collector.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Collector Feedback</h1><p>Please ensure static/collector.html exists</p>", status_code=404)

@app.get("/ops", response_class=HTMLResponse)
async def ops_page():
    try:
        with open("static/ops.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Operations Dashboard</h1><p>Please ensure static/ops.html exists</p>", status_code=404)

@app.get("/credits", response_class=HTMLResponse)
async def credits_page():
    try:
        with open("static/credits.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Credit Dashboard</h1><p>Please ensure static/credits.html exists</p>", status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
