import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
from ocr_service import OCRService
import logging
import plotly.graph_objects as go
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Insurance Claim OCR Demo",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2a5298;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .field-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ocr_service' not in st.session_state:
    st.session_state.ocr_service = None
    st.session_state.results = None
    st.session_state.processing_time = 0

def load_ocr_service():
    """Initialize OCR service with loading spinner"""
    if st.session_state.ocr_service is None:
        with st.spinner("Initializing OCR engines... This may take a moment for the first run."):
            st.session_state.ocr_service = OCRService()
    return st.session_state.ocr_service

def create_confidence_chart(results):
    """Create a confidence comparison chart"""
    methods = []
    confidences = []
    
    for method, data in results['all_attempts'].items():
        methods.append(method.upper())
        confidences.append(data['confidence'] * 100)  # Convert to percentage
    
    fig = go.Figure(data=[
        go.Bar(
            x=methods,
            y=confidences,
            marker_color=['#28a745' if method == results['method_used'].upper() else '#6c757d' for method in methods],
            text=[f"{conf:.1f}%" for conf in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="OCR Method Confidence Comparison",
        xaxis_title="OCR Method",
        yaxis_title="Confidence (%)",
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def display_extracted_fields(fields):
    """Display extracted insurance fields in a nice format"""
    st.subheader("📋 Extracted Insurance Information")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        if fields['policy_number']:
            st.markdown(f"""
            <div class="field-card">
                <strong>📝 Policy Number:</strong><br>
                <code>{fields['policy_number']}</code>
            </div>
            """, unsafe_allow_html=True)
        
        if fields['claim_number']:
            st.markdown(f"""
            <div class="field-card">
                <strong>🎫 Claim Number:</strong><br>
                <code>{fields['claim_number']}</code>
            </div>
            """, unsafe_allow_html=True)
        
        if fields['insured_name']:
            st.markdown(f"""
            <div class="field-card">
                <strong>👤 Insured Name:</strong><br>
                {fields['insured_name']}
            </div>
            """, unsafe_allow_html=True)
        
        if fields['date_of_loss']:
            st.markdown(f"""
            <div class="field-card">
                <strong>📅 Date of Loss:</strong><br>
                {fields['date_of_loss']}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if fields['claim_amount']:
            st.markdown(f"""
            <div class="field-card">
                <strong>💰 Claim Amount:</strong><br>
                <span style="color: #28a745; font-weight: bold;">$${fields['claim_amount']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        if fields['phone_number']:
            st.markdown(f"""
            <div class="field-card">
                <strong>📞 Phone Number:</strong><br>
                {fields['phone_number']}
            </div>
            """, unsafe_allow_html=True)
        
        if fields['email']:
            st.markdown(f"""
            <div class="field-card">
                <strong>📧 Email:</strong><br>
                {fields['email']}
            </div>
            """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏢 Insurance Claim OCR Demo</h1>
        <p>AI-Powered Document Processing for Insurance Claims</p>
        <p><strong>Multi-Model OCR:</strong> EasyOCR → Tesseract → PaddleOCR</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ How it Works")
        st.info("""
        **Hierarchical OCR Processing:**
        
        1. **EasyOCR** (Threshold: 70%)
           - Fast & accurate for most documents
        
        2. **Tesseract** (Threshold: 60%)
           - Falls back if EasyOCR confidence < 70%
        
        3. **PaddleOCR** (Threshold: 50%)
           - Final attempt for difficult documents
        
        **Supported Formats:**
        - PDF files
        - JPG/JPEG images
        - PNG images
        """)
        
        st.header("🎯 Demo Features")
        st.success("""
        ✅ Multi-model OCR processing
        
        ✅ Automatic field extraction
        
        ✅ Confidence scoring
        
        ✅ Real-time processing
        
        ✅ Insurance-specific parsing
        """)
    
    # Main content
    uploaded_file = st.file_uploader(
        "📁 Upload Insurance Claim Form",
        type=['pdf', 'jpg', 'jpeg', 'png'],
        help="Upload a PDF or image file of an insurance claim form"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB",
            "File type": uploaded_file.type
        }
        
        st.markdown("### 📋 File Information")
        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Filename", file_details["Filename"])
        col2.metric("📊 File Size", file_details["File size"])
        col3.metric("🔖 File Type", file_details["File type"])
        
        # Convert uploaded file to image
        try:
            if uploaded_file.type == "application/pdf":
                st.warning("⚠️ PDF support requires additional setup. Please convert to image format.")
                return
            
            # Load image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Display original image
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("### 🖼️ Original Document")
                st.image(image, use_column_width=True)
            
            with col2:
                st.markdown("### ⚙️ Processing Status")
                
                if st.button("🚀 Start OCR Processing", type="primary"):
                    # Initialize OCR service
                    ocr_service = load_ocr_service()
                    
                    # Create progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Start processing
                    start_time = time.time()
                    
                    status_text.text("🔄 Processing with hierarchical OCR...")
                    progress_bar.progress(25)
                    
                    # Process with OCR
                    results = ocr_service.hierarchical_ocr(image_array)
                    progress_bar.progress(75)
                    
                    # Extract insurance fields
                    status_text.text("📊 Extracting insurance fields...")
                    fields = ocr_service.extract_insurance_fields(results['final_text'])
                    progress_bar.progress(100)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    status_text.success(f"✅ Processing complete! ({processing_time:.2f}s)")
                    
                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.fields = fields
                    st.session_state.processing_time = processing_time
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return
    
    # Display results if available
    if st.session_state.results is not None:
        st.markdown("---")
        st.markdown("## 📈 Processing Results")
        
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 OCR Method Used", 
                st.session_state.results['method_used'],
                help="The OCR method that provided the best results"
            )
        
        with col2:
            confidence_pct = st.session_state.results['final_confidence'] * 100
            st.metric(
                "📊 Confidence Score", 
                f"{confidence_pct:.1f}%",
                help="Overall confidence in the OCR results"
            )
        
        with col3:
            st.metric(
                "⏱️ Processing Time", 
                f"{st.session_state.processing_time:.2f}s",
                help="Total time taken for processing"
            )
        
        with col4:
            text_length = len(st.session_state.results['final_text'])
            st.metric(
                "📝 Text Length", 
                f"{text_length} chars",
                help="Number of characters extracted"
            )
        
        # Display confidence chart
        st.plotly_chart(
            create_confidence_chart(st.session_state.results), 
            use_container_width=True
        )
        
        # Display extracted fields
        if hasattr(st.session_state, 'fields'):
            display_extracted_fields(st.session_state.fields)
        
        # Display raw extracted text
        with st.expander("📄 Raw Extracted Text", expanded=False):
            st.text_area(
                "Extracted Text:",
                value=st.session_state.results['final_text'],
                height=200,
                disabled=True
            )
        
        # Display detailed results
        with st.expander("🔍 Detailed OCR Results", expanded=False):
            for method, data in st.session_state.results['all_attempts'].items():
                if data['text']:  # Only show methods that returned text
                    st.markdown(f"**{method.upper()}** (Confidence: {data['confidence']:.2%})")
                    st.text(data['text'][:500] + "..." if len(data['text']) > 500 else data['text'])
                    st.markdown("---")
        
        # Export options
        st.markdown("### 📤 Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as JSON
            import json
            export_data = {
                'extracted_fields': st.session_state.fields if hasattr(st.session_state, 'fields') else {},
                'ocr_results': {
                    'method_used': st.session_state.results['method_used'],
                    'confidence': st.session_state.results['final_confidence'],
                    'text': st.session_state.results['final_text']
                },
                'processing_time': st.session_state.processing_time
            }
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="📋 Download as JSON",
                data=json_str,
                file_name="ocr_results.json",
                mime="application/json"
            )
        
        with col2:
            # Export as CSV
            if hasattr(st.session_state, 'fields'):
                df = pd.DataFrame([st.session_state.fields])
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download as CSV",
                    data=csv,
                    file_name="extracted_fields.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()