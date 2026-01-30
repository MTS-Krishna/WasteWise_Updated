# WasteWise User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Using the Main Dashboard](#using-the-main-dashboard)
4. [Collector Feedback System](#collector-feedback-system)
5. [Operations Dashboard](#operations-dashboard)
6. [Credit System](#credit-system)
7. [QR Code System](#qr-code-system)
8. [Troubleshooting](#troubleshooting)

## Introduction

WasteWise is an intelligent waste classification system that helps you properly sort and dispose of waste items. The system uses AI to analyze receipts, shopping lists, or menus and automatically classifies items into appropriate waste categories.

## Getting Started

### First Time Setup

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Server**
   ```bash
   python app.py
   ```
   You should see: `Uvicorn running on http://127.0.0.1:8000`

3. **Open Your Browser**
   Navigate to: `http://127.0.0.1:8000`

## Using the Main Dashboard

### Uploading Files

1. **Select a Bin**: Choose which waste bin this upload is for (bin-A, bin-B, bin-C, or bin-D)

2. **Upload a File**:
   - Click the upload area or drag and drop a file
   - Supported formats: JPG, PNG, PDF, DOCX, TXT, CSV, JSON
   - Common sources: Grocery receipts, restaurant menus, shopping lists

3. **Process the File**:
   - Click "Process File & Classify Items"
   - Wait for the AI to analyze and classify items
   - View results with animations and visualizations

### Understanding Results

**Classification Table** shows:
- **Item**: The waste item name
- **Category**: PET, Glass, Paper, Metal, MLP, Compost, or Other
- **Stream**: Dry, Wet, Recyclable, or None
- **Recyclability**: High, Moderate, Low, or None
- **Weight**: Estimated weight in kilograms
- **Note**: Specific disposal instructions

**Charts** display:
- Recyclability breakdown (pie chart)
- Waste stream distribution (bar chart)

**Bag Recipes** provide:
- Grouped items by waste stream
- Number of bags needed
- Detailed disposal instructions

### Weight Calculation

The system automatically estimates weights by:
- Extracting quantities (e.g., "2 lbs", "1 gallon", "1 dozen")
- Converting units (lbs → kg, gallons → kg)
- Using smart defaults based on item type
- Providing accurate total weights

## Collector Feedback System

### Using QR Codes

1. **Generate QR Codes** (Admin):
   ```bash
   python generate_qrs.py
   ```
   QR codes are saved in the `bin_qrs/` folder

2. **Print and Attach**: Print QR codes and attach them to physical bins

3. **Scan QR Code**: Collectors scan the QR code with their phone

4. **Provide Feedback**:
   - Click "✅ Bin is Valid" if contents are correct
   - Click "⚠️ Bin is Contaminated" if there are issues
   - Feedback is automatically saved

### Direct Access

You can also access the collector page directly:
```
http://127.0.0.1:8000/collector?bin_id=bin-A
```

## Operations Dashboard

### Viewing Bin Status

The operations dashboard shows:
- **Fill Levels**: Visual progress bars for each bin
- **Locations**: GPS coordinates of each bin
- **Capacity**: Current fill vs. total capacity

### Route Optimization

1. Click "Optimize Route Now"
2. System finds bins ≥75% full
3. Calculates optimal collection route
4. Displays route on map with distance

### Analytics

- **Total Submissions**: Number of feedback entries
- **Valid Bags**: Count of valid classifications
- **Contaminated Bags**: Count of contamination reports
- **Contamination Rate**: Percentage of contaminated items
- **Status Distribution**: Pie chart of feedback status

## Credit System

### Depositing Recyclable Plastics

1. Navigate to Credit System page
2. Enter your User ID
3. Enter weight of plastic (in kg)
4. Click "Deposit Plastic"
5. Earn 1 credit per kg deposited

### Checking Balance

1. Enter your User ID
2. Click "Check Balance"
3. View your current credit balance

### Credit Rules

- Only "Recyclable Plastics" are accepted
- Rate: 1 credit per kg
- Credits are stored permanently
- Balance persists across sessions

## QR Code System

### Generating QR Codes

1. **Run the Generator**:
   ```bash
   python generate_qrs.py
   ```

2. **Output**: QR codes saved in `bin_qrs/` folder
   - `qr_code_bin-A.png`
   - `qr_code_bin-B.png`
   - `qr_code_bin-C.png`
   - `qr_code_bin-D.png`

3. **Customization**: Edit `generate_qrs.py` to:
   - Change URL template for production
   - Add more bins
   - Customize QR code styling

### QR Code URLs

Each QR code contains a URL like:
```
http://127.0.0.1:8000/collector?bin_id=bin-A
```

For production, update the URL template in `generate_qrs.py`.

## Troubleshooting

### File Upload Issues

**Problem**: "No text found in file"
- **Solution**: Ensure file is clear and readable
- Try a different file format
- Check file isn't corrupted

**Problem**: "No valid items found"
- **Solution**: File may contain only prices/numbers
- Try a file with item names
- Check OCR is working correctly

### Classification Issues

**Problem**: Items showing as "Unknown"
- **Solution**: This is normal for uncommon items
- System uses fallback classification
- Items are still categorized appropriately

**Problem**: Weights seem incorrect
- **Solution**: System estimates based on item names
- Check if quantities are in the item name
- Manual adjustment may be needed for special cases

### Server Issues

**Problem**: Server won't start
- **Solution**: Check port 8000 is available
- Verify all dependencies are installed
- Check Python version (3.8+)

**Problem**: AI classification not working
- **Solution**: Verify Ollama is running
- Check model is available: `ollama list`
- System will use fallback if AI fails

### QR Code Issues

**Problem**: QR codes don't work
- **Solution**: Verify server is running
- Check URL template is correct
- Ensure collector endpoint is accessible

## Tips and Best Practices

### For Best Classification Results

1. **Clear Images**: Use high-quality, well-lit photos
2. **Complete Receipts**: Include all items, not just partial lists
3. **Standard Formats**: Use common file formats (PDF, JPG, PNG)
4. **Item Names**: Ensure item names are readable in the source

### For Accurate Weights

1. **Include Quantities**: Items with quantities get better weight estimates
   - Good: "2 lbs chicken", "1 gallon milk"
   - Less accurate: "chicken", "milk"

2. **Use Standard Units**: System recognizes lbs, kg, gallons, liters, etc.

3. **Multiple Items**: System handles plural items and counts

### For Operations

1. **Regular Monitoring**: Check operations dashboard regularly
2. **Route Optimization**: Run optimization when bins are getting full
3. **Feedback Collection**: Encourage collectors to use QR codes
4. **Data Backup**: Regularly backup JSON database files

## Advanced Features

### Knowledge Graph

Edit `pack_graph.json` to add known items:
```json
{
  "item_name": {
    "category": "PET",
    "stream": "Recyclable",
    "recyclability": "High",
    "note": "Rinse before disposal.",
    "weight_kg": 0.05
  }
}
```

### Custom Bin Configuration

Edit `app.py` to add/modify bins:
```python
bin_data = {
    "bin-E": {
        "capacity_kg": 30.0,
        "fill_level_kg": 0.0,
        "location": (lat, lng)
    }
}
```

### API Integration

All features are available via REST API:
- Use `POST /process_file` for programmatic classification
- Use `GET /analytics` for data integration
- Use `POST /bin_feedback` for automated feedback

## Support

For additional help:
- Check the README.md for technical details
- Review API documentation in app.py
- Open an issue on the repository

---

**Happy Recycling! ♻️**

