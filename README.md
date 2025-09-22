# Insurance Claim OCR Demo

AI-Powered document processing demo for insurance claim forms using hierarchical OCR approach with multiple models.

## Project Overview

This project demonstrates an intelligent OCR system that processes insurance claim forms using a multi-tiered approach:

1. **EasyOCR** (Primary - Threshold: 70%)
2. **Tesseract** (Fallback - Threshold: 60%)
3. **PaddleOCR** (Final attempt - Threshold: 50%)

The system automatically selects the best OCR method based on confidence scores and extracts key insurance information.

## Features

- **Multi-Model OCR Processing**: Hierarchical approach ensuring maximum accuracy
- **Insurance Field Extraction**: Automatic parsing of policy numbers, claim amounts, dates, etc.
- **Real-time Processing**: Fast document analysis with progress tracking
- **Interactive UI**: Clean, professional interface built with Streamlit
- **Export Options**: Download results as JSON or CSV
- **Confidence Scoring**: Visual confidence comparison between OCR methods

## Project Structure

```
insurance-ocr-demo/
├── app.py                 # Main Streamlit application
├── ocr_service.py         # Backend OCR processing service
├── requirements.txt       # Python dependencies
├── packages.txt          # System packages for Streamlit Cloud
├── Dockerfile            # Docker configuration
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── README.md             # This file
```

## Installation & Setup

### Method 1: Local Development

1. **Clone the repository** (or create files manually):
```bash
git clone <your-repo-url>
cd insurance-ocr-demo
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install system dependencies** (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

5. **Run the application**:
```bash
streamlit run app.py
```

### Method 2: Docker

1. **Build Docker image**:
```bash
docker build -t insurance-ocr-demo .
```

2. **Run container**:
```bash
docker run -p 8501:8501 insurance-ocr-demo
```

## Deployment Guide

### Deploy on Streamlit Cloud (Recommended - FREE)

1. **Prepare your GitHub repository**:
   - Create a new repository on GitHub
   - Upload all files from this project
   - Ensure `requirements.txt` and `packages.txt` are in the root directory

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path to `app.py`
   - Click "Deploy"

3. **Access your app**:
   - Your app will be available at `https://[your-app-name].streamlit.app`
   - Sharing link will be provided for the assignment

### Alternative: Deploy on Heroku

1. **Create Heroku app**:
```bash
heroku create your-insurance-ocr-demo
```

2. **Add buildpacks**:
```bash
heroku buildpacks:add --index 1 heroku-community/apt
heroku buildpacks:add --index 2 heroku/python
```

3. **Create Aptfile** (for system dependencies):
```bash
echo "tesseract-ocr tesseract-ocr-eng" > Aptfile
```

4. **Deploy**:
```bash
git push heroku main
```

### Alternative: Deploy on Railway

1. **Connect GitHub repository** to Railway
2. **Set build command**: `pip install -r requirements.txt`
3. **Set start command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Usage

1. **Upload Document**: 
   - Supported formats: JPG, JPEG, PNG
   - Maximum size: 200MB

2. **Process Document**:
   - Click "Start OCR Processing"
   - Watch real-time progress
   - View confidence scores for each OCR method

3. **Review Results**:
   - Extracted insurance fields
   - Raw text output
   - Confidence comparison chart
   - Processing time metrics

4. **Export Data**:
   - Download as JSON for system integration
   - Download as CSV for spreadsheet analysis

## Technical Details

### OCR Processing Pipeline

1. **Image Preprocessing**:
   - Grayscale conversion
   - Noise reduction
   - Adaptive thresholding

2. **Hierarchical OCR**:
   - **EasyOCR**: Fast, neural network-based
   - **Tesseract**: Traditional, highly configurable
   - **PaddleOCR**: Advanced, handles complex layouts

3. **Field Extraction**:
   - Regex patterns for insurance-specific fields
   - Policy numbers, claim amounts, dates, contact info

### Key Dependencies

- **Streamlit**: Web interface framework
- **OpenCV**: Image processing
- **EasyOCR**: Primary OCR engine
- **Tesseract**: Fallback OCR engine
- **PaddleOCR**: Final fallback OCR engine
- **Plotly**: Interactive charts

## Performance Metrics

The system tracks several performance indicators:
- **Processing Time**: End-to-end processing duration
- **Confidence Scores**: OCR accuracy for each method
- **Field Extraction Rate**: Percentage of insurance fields successfully identified
- **Method Selection**: Which OCR engine provided the best results

## AI Prompts Used in Development

Here are 5 key AI prompts used to build this system:

1. **"Create a hierarchical OCR system using EasyOCR, Tesseract, and PaddleOCR with confidence thresholds"**
2. **"Design a Streamlit UI for insurance document processing with professional styling and real-time progress"**
3. **"Write regex patterns to extract insurance claim form fields like policy numbers, claim amounts, and dates"**
4. **"Create a deployment configuration for Streamlit Cloud with all necessary dependencies"**
5. **"Build error handling and logging system for multi-model OCR processing pipeline"**

## Demo Features for Business Audience

- **Professional Interface**: Clean, corporate-friendly design
- **Real-time Processing**: No waiting - see results as they're processed
- **Confidence Visualization**: Clear metrics showing system reliability
- **Export Ready**: Data in formats ready for business systems
- **Error Handling**: Graceful fallbacks when one OCR method fails

## Future Enhancements

- PDF support with text layer detection
- Batch processing for multiple documents
- Machine learning model training for custom insurance forms
- Integration with insurance management systems
- Multi-language support

## Production Considerations

**Note**: This is a demo application. For production use, consider:
- Authentication and authorization
- Database integration for storing results
- API rate limiting
- Enhanced error handling
- Security scanning of uploaded files
- Scalability improvements

## Support

For technical questions or deployment issues:
- Check the [Streamlit documentation](https://docs.streamlit.io)
- Review [EasyOCR documentation](https://github.com/JaidedAI/EasyOCR)
- Open an issue in the GitHub repository

## License

This project is created for demonstration purposes as part of an AI Engineering Intern assignment.

---