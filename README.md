# WasteWise ♻️ - Smart Waste Classification System

A modern, AI-powered waste classification system that helps users properly sort and dispose of waste items. The system uses advanced machine learning and a knowledge graph to classify waste items and generate smart disposal instructions.

## 🌟 Features

- **AI-Powered Classification**: Uses LLM (Large Language Models) to intelligently classify waste items
- **Smart Weight Estimation**: Automatically estimates weights based on item names, quantities, and units
- **Knowledge Graph**: Pre-configured database of common items for instant classification
- **Modern Web Interface**: Beautiful, responsive HTML/CSS/JavaScript frontend with advanced animations
- **Bin Management**: Track multiple waste bins with fill levels and locations
- **Route Optimization**: TSP-based route optimization for waste collection
- **Credit System**: Reward users for recycling recyclable plastics
- **QR Code Integration**: Generate QR codes for bins to enable quick feedback collection

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR (for image text extraction)
- Ollama or compatible LLM API (for AI classification)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd WasteWise
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Tesseract OCR**
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki) and install
   - The path is configured in `app.py` (line 24)
   - Update the path if your installation differs

4. **Set up Ollama (for AI classification)**
   - Install Ollama from [ollama.ai](https://ollama.ai)
   - Pull a model: `ollama pull llama3.2` or `ollama pull mistral`
   - The system will try multiple models automatically

5. **Start the FastAPI server**
   ```bash
   python app.py
   ```
   The server will start on `http://127.0.0.1:8000`

6. **Access the application**
   - Main Dashboard: `http://127.0.0.1:8000/`
   - Collector Feedback: `http://127.0.0.1:8000/collector?bin_id=bin-A`
   - Operations Dashboard: `http://127.0.0.1:8000/ops`
   - Credit System: `http://127.0.0.1:8000/credits`

## 📁 Project Structure

```
WasteWise/
├── app.py                 # FastAPI backend server
├── generate_qrs.py        # QR code generator for bins
├── requirements.txt       # Python dependencies
├── pack_graph.json        # Knowledge graph of known items
├── classification_db.json  # Database of classified items
├── credits_db.json        # User credit balances
├── static/                # Frontend HTML files
│   ├── index.html         # Main waste classification dashboard
│   ├── collector.html     # Bin feedback collector interface
│   ├── ops.html           # Operations dashboard
│   └── credits.html       # Credit system dashboard
├── bin_qrs/               # Generated QR codes for bins
│   ├── qr_code_bin-A.png
│   ├── qr_code_bin-B.png
│   ├── qr_code_bin-C.png
│   └── qr_code_bin-D.png
└── Guide/                 # Documentation
    ├── User_Guide.pdf
    └── WasteWise.pptx
```

## 🎯 Usage Guide

### 1. Classifying Waste Items

1. Navigate to the main dashboard (`http://127.0.0.1:8000/`)
2. Select a target waste bin (bin-A, bin-B, bin-C, or bin-D)
3. Upload a file containing waste items:
   - **Supported formats**: JPG, PNG, PDF, DOCX, TXT, CSV, JSON
   - **Sources**: Receipts, menus, shopping lists, etc.
4. Click "Process File & Classify Items"
5. View the classification results:
   - Item categories (PET, Glass, Paper, Metal, MLP, Compost, Other)
   - Waste streams (Dry, Wet, Recyclable, None)
   - Recyclability ratings (High, Moderate, Low, None)
   - Estimated weights
   - Disposal instructions

### 2. Using QR Codes for Bin Feedback

1. Generate QR codes:
   ```bash
   python generate_qrs.py
   ```
2. Print and attach QR codes to physical bins
3. Collectors scan QR codes to provide feedback
4. Feedback updates bin fill levels automatically

### 3. Operations Dashboard

- View real-time bin status and fill levels
- Monitor feedback analytics
- Optimize collection routes for bins ≥75% full
- Track contamination rates

### 4. Credit System

- Users deposit recyclable plastics
- Earn 1 credit per kg of plastic
- Check credit balances
- Credits are automatically calculated and stored

## 🔧 Configuration

### Backend Configuration

**Ollama API URL** (app.py, line 33):
```python
OLLAMA_API_URL = "http://localhost:11434/api/generate"
```

**Tesseract Path** (app.py, line 24):
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**Bin Configuration** (app.py, lines 46-51):
```python
bin_data = {
    "bin-A": {"capacity_kg": 25.0, "fill_level_kg": 0.0, "location": (40.71, -74.00)},
    "bin-B": {"capacity_kg": 25.0, "fill_level_kg": 0.0, "location": (34.05, -118.24)},
    # ... more bins
}
```

### QR Code Configuration

**URL Template** (generate_qrs.py, line 19):
```python
COLLECTOR_APP_URL_TEMPLATE = "http://127.0.0.1:8000/collector?bin_id={}"
```

For production, update to your domain:
```python
COLLECTOR_APP_URL_TEMPLATE = "https://yourdomain.com/collector?bin_id={}"
```

## 🧠 How It Works

### Classification Process

1. **Text Extraction**: OCR extracts text from images/PDFs
2. **Item Cleaning**: Removes prices, quantities, and formatting
3. **Knowledge Graph Lookup**: Checks pre-configured items first
4. **AI Classification**: Uses LLM to classify unknown items
5. **Weight Estimation**: Smart algorithm estimates weights from item names
6. **Fallback Classification**: Keyword-based classification if AI fails

### Weight Estimation Algorithm

The system intelligently estimates weights by:
- Extracting quantities and units (lbs, kg, gallons, dozen, etc.)
- Converting units automatically (lbs → kg, gallons → kg)
- Using category-based defaults (meat ≈0.7kg, fruits ≈0.15kg each)
- Handling plural items and multiple quantities

### API Endpoints

- `POST /process_file` - Upload and classify waste items
- `POST /bin_feedback` - Submit bin feedback
- `GET /analytics` - Get analytics data
- `GET /optimize_routes` - Optimize collection routes
- `POST /deposit_recyclable` - Deposit recyclable plastics
- `GET /user_balance/{user_id}` - Get user credit balance

## 🎨 UI Features

- **Animated Background**: Floating particle effects
- **Smooth Animations**: Fade-in, slide, and scale transitions
- **Interactive Charts**: Real-time data visualization
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern Design**: Gradient backgrounds, shadows, and 3D effects
- **Micro-interactions**: Hover effects and button animations

## 📊 Data Storage

- **classification_db.json**: Stores all classification history
- **credits_db.json**: Stores user credit balances
- **pack_graph.json**: Knowledge graph of known items
- All data is stored in JSON format for easy backup and migration

## 🔒 Security Notes

- Currently uses in-memory storage (data persists in JSON files)
- For production, consider:
  - Database integration (PostgreSQL, MongoDB)
  - Authentication and authorization
  - HTTPS/SSL certificates
  - Input validation and sanitization
  - Rate limiting

## 🐛 Troubleshooting

### OCR Not Working
- Verify Tesseract is installed and path is correct
- Check file permissions
- Ensure images are clear and readable

### AI Classification Failing
- Verify Ollama is running: `ollama list`
- Check available models: `ollama list`
- The system automatically tries multiple models
- Fallback classification ensures items are always classified

### QR Codes Not Working
- Verify FastAPI server is running
- Check URL template in `generate_qrs.py`
- Ensure collector endpoint is accessible

## 🚀 Production Deployment

1. **Update URLs**: Change localhost URLs to production domain
2. **Database**: Migrate from JSON to proper database
3. **Security**: Add authentication, HTTPS, and input validation
4. **Scaling**: Use production ASGI server (Gunicorn + Uvicorn)
5. **Monitoring**: Add logging and error tracking

## 📝 License

MIT License

## 📧 Support

For issues and questions, please open an issue on the repository.

---

**Made with ♻️ for a cleaner planet**

