import streamlit as st
from pdf2image import convert_from_bytes
import cv2
import numpy as np
from PIL import Image
import io
import time
import logging
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json

# Import custom modules
try:
    from ocr_service import OCRService
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure ocr_service.py is in the same directory.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Insurance Form ROI-OCR Demo",
    page_icon="📋",
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
    
    .field-success {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    
    .field-warning {
        background: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    
    .field-error {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
    
    .roi-info {
        background: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #b3d9ff;
        margin: 1rem 0;
    }
    
    .method-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    .badge-easyocr { background: #28a745; color: white; }
    .badge-tesseract { background: #17a2b8; color: white; }
    .badge-paddleocr { background: #fd7e14; color: white; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'ocr_service' not in st.session_state:
        st.session_state.ocr_service = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = 0

def load_ocr_service():
    """Initialize OCR service with loading spinner"""
    try:
        if st.session_state.ocr_service is None:
            with st.spinner("🔧 Initializing OCR engines... This may take a moment for the first run."):
                st.session_state.ocr_service = OCRService()
        return st.session_state.ocr_service
    except Exception as e:
        st.error(f"Failed to initialize OCR service: {e}")
        logger.error(f"OCR service initialization failed: {e}")
        return None

def process_uploaded_file(uploaded_file):
    """Process uploaded image file"""
    try:
        # Handle image files only
        image = Image.open(uploaded_file)
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)
        return image_array, image
    except Exception as e:
        st.error(f"❌ Error loading image: {str(e)}")
        logger.error(f"Image processing error: {e}")
        return None, None

def create_roi_visualization(image_shape):
    """Create visualization of ROI regions"""
    try:
        h, w = image_shape[:2]
        
        # Create figure
        fig = go.Figure()
        
        # Add image background (placeholder)
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=w, y1=h,
            line=dict(color="lightgray", width=2),
            fillcolor="lightgray",
            opacity=0.3
        )
        
        # Add ROI rectangles
        if st.session_state.ocr_service and hasattr(st.session_state.ocr_service, 'roi_detector'):
            roi_detector = st.session_state.ocr_service.roi_detector
            colors = px.colors.qualitative.Set3
            
            for i, (field_name, field_config) in enumerate(roi_detector.FIELD_DEFINITIONS.items()):
                roi = field_config['roi']
                x, y, roi_w, roi_h = roi
                
                # Convert to pixel coordinates
                x1, y1 = int(x * w), int(y * h)
                x2, y2 = int((x + roi_w) * w), int((y + roi_h) * h)
                
                color = colors[i % len(colors)]
                
                fig.add_shape(
                    type="rect",
                    x0=x1, y0=y1, x1=x2, y1=y2,
                    line=dict(color=color, width=2),
                    fillcolor=color,
                    opacity=0.2
                )
                
                # Add field label
                fig.add_annotation(
                    x=x1 + (x2-x1)/2,
                    y=y1 + (y2-y1)/2,
                    text=field_name.replace('_', ' ').title(),
                    showarrow=False,
                    font=dict(size=8, color="black"),
                    bgcolor="white",
                    opacity=0.8
                )
        
        fig.update_layout(
            title="ROI Field Detection Map",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig
    except Exception as e:
        logger.error(f"ROI visualization error: {e}")
        return None

def create_confidence_chart(results):
    """Create confidence chart for processed fields"""
    try:
        fields = []
        confidences = []
        methods = []
        
        for field_name, field_data in results['processed_fields'].items():
            if field_data['text']:  # Only include fields with extracted text
                fields.append(field_name.replace('_', ' ').title())
                confidences.append(field_data['confidence'] * 100)
                methods.append(field_data['method_used'])
        
        if not fields:
            return None
        
        # Color mapping for methods
        color_map = {
            'EasyOCR': '#28a745',
            'Tesseract (PSM 7)': '#17a2b8', 
            'Tesseract (PSM 8)': '#17a2b8',
            'PaddleOCR': '#fd7e14'
        }
        
        colors = [color_map.get(method, '#6c757d') for method in methods]
        
        fig = go.Figure(data=[
            go.Bar(
                x=fields,
                y=confidences,
                marker_color=colors,
                text=[f"{conf:.1f}%" for conf in confidences],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<br>Method: %{customdata}<extra></extra>',
                customdata=methods
            )
        ])
        
        fig.update_layout(
            title="Field Extraction Confidence by ROI",
            xaxis_title="Insurance Form Fields",
            yaxis_title="Confidence (%)",
            height=500,
            yaxis=dict(range=[0, 100]),
            xaxis=dict(tickangle=45)
        )
        
        return fig
    except Exception as e:
        logger.error(f"Confidence chart creation error: {e}")
        return None

def display_field_results(results):
    """Display extracted field results in organized format"""
    try:
        st.subheader("📋 Extracted Insurance Form Fields")
        
        # Create tabs for different categories
        tab1, tab2, tab3, tab4 = st.tabs(["👤 Personal Info", "🏥 Medical Info", "💰 Financial Info", "📊 All Fields"])
        
        personal_fields = ['name', 'hkid', 'passport_number', 'date_of_birth', 'sex', 'address', 'occupation']
        medical_fields = ['patient_name', 'doctors_name', 'treatment_date', 'date_of_admission', 'date_of_discharge', 
                         'consultation_date', 'symptoms_duration', 'symptoms_complaints', 'hospitalization_details', 
                         'final_diagnosis', 'operation_date', 'operation_procedure']
        financial_fields = ['account_holder_name', 'currency', 'bank_name', 'hkd_bank_name', 'bank_no', 'branch_no', 'bank_account_no']
        insurance_fields = ['policy_number', 'cert_number', 'name_of_policyowner', 'employee_member', 'name_of_insured']
        
        def display_field_category(field_list, title):
            st.markdown(f"### {title}")
            cols = st.columns(2)
            col_idx = 0
            
            for field_name in field_list:
                if field_name in results['processed_fields']:
                    field_data = results['processed_fields'][field_name]
                    
                    with cols[col_idx % 2]:
                        display_single_field(field_name, field_data)
                        col_idx += 1
        
        def display_single_field(field_name, field_data):
            field_title = field_name.replace('_', ' ').title()
            
            if field_data['text']:
                confidence = field_data['confidence']
                method = field_data['method_used']
                final_text = field_data.get('final_text', field_data['text'])
                validation = field_data.get('validation', {})
                
                # Determine field status
                if confidence >= 0.7 and validation.get('is_valid', False):
                    css_class = "field-success"
                    status_icon = "✅"
                elif confidence >= 0.5:
                    css_class = "field-warning" 
                    status_icon = "⚠️"
                else:
                    css_class = "field-error"
                    status_icon = "❌"
                
                # Method badge
                method_class = "badge-easyocr" if "EasyOCR" in method else "badge-tesseract" if "Tesseract" in method else "badge-paddleocr"
                
                st.markdown(f"""
                <div class="{css_class}">
                    <strong>{status_icon} {field_title}</strong>
                    <span class="method-badge {method_class}">{method}</span><br>
                    <code style="font-size: 1.1em;">{final_text}</code><br>
                    <small>Confidence: {confidence:.1%} | ROI: {field_data['roi_size']}</small>
                    {f"<br><small style='color: orange;'>⚠️ {', '.join(validation.get('validation_notes', []))}</small>" if validation.get('validation_notes') else ""}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="field-error">
                    <strong>❌ {field_title}</strong><br>
                    <em>No text extracted</em><br>
                    <small>ROI size: {field_data['roi_size']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Display by category
        with tab1:
            display_field_category(personal_fields + insurance_fields, "Personal & Insurance Information")
        
        with tab2:
            display_field_category(medical_fields, "Medical Information")
        
        with tab3:
            display_field_category(financial_fields, "Financial Information")
        
        with tab4:
            st.markdown("### All Processed Fields")
            for field_name, field_data in results['processed_fields'].items():
                with st.expander(f"📄 {field_name.replace('_', ' ').title()}", expanded=False):
                    display_single_field(field_name, field_data)
                    
                    # Show all OCR attempts
                    if 'attempts' in field_data:
                        st.markdown("**All OCR Attempts:**")
                        for method, attempt in field_data['attempts'].items():
                            if attempt['text']:
                                st.markdown(f"- **{method.upper()}**: {attempt['text']} (Confidence: {attempt['confidence']:.1%})")
    except Exception as e:
        st.error(f"Error displaying field results: {e}")
        logger.error(f"Field results display error: {e}")

def create_summary_metrics(results):
    """Create summary metrics display"""
    try:
        summary = results['summary']
        total_fields = results['total_fields']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Total Fields", 
                total_fields,
                help="Total number of defined ROI fields"
            )
        
        with col2:
            success_rate = summary['extraction_rate'] * 100
            st.metric(
                "✅ Extraction Rate", 
                f"{success_rate:.1f}%",
                help="Percentage of fields with extracted text"
            )
        
        with col3:
            validation_rate = summary['validation_rate'] * 100
            st.metric(
                "🎯 Validation Rate", 
                f"{validation_rate:.1f}%",
                help="Percentage of fields passing validation"
            )
        
        with col4:
            high_conf_rate = (summary['high_confidence_fields'] / total_fields) * 100
            st.metric(
                "🏆 High Confidence", 
                f"{high_conf_rate:.1f}%",
                help="Percentage of fields with >70% confidence"
            )
    except Exception as e:
        st.error(f"Error creating summary metrics: {e}")
        logger.error(f"Summary metrics error: {e}")

from pdf2image import convert_from_bytes

def main():
    """Main application function"""
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📋 Insurance Form ROI-OCR Demo</h1>
        <p>Advanced Field-Specific OCR Processing with ROI Detection</p>
        <p><strong>Hierarchical OCR:</strong> EasyOCR → Tesseract (PSM 7/8) → PaddleOCR</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 ROI-Based Processing")
        st.info("""
        **Field-Specific OCR Pipeline:**
        
        1. **ROI Extraction**
           - 40+ predefined form fields
           - Normalized coordinates (0-1)
           - Field-specific preprocessing
        
        2. **Hierarchical OCR**
           - EasyOCR (primary)
           - Tesseract PSM 7/8 (fallback)
           - PaddleOCR (final attempt)
        
        3. **Smart Validation**
           - HKID format checking
           - Date pattern validation
           - Currency recognition
        """)
        
        st.header("📋 Supported Formats")
        st.success("""
        **Image Files & PDF:**
        • JPG/JPEG  
        • PNG  
        • BMP  
        • TIFF  
        • PDF (auto convert first page to image)  
        """)
    
    # Main content
    uploaded_file = st.file_uploader(
        "📁 Upload Insurance Form Image or PDF",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'pdf'],
        help="Upload a JPG, PNG, BMP, TIFF image or PDF of an insurance form"
    )

    if uploaded_file is not None:
        # Display basic file info
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
        
        # Process uploaded file
        try:
            if uploaded_file.type == "application/pdf":
                pdf_pages = convert_from_bytes(uploaded_file.read())
                images = []
                for i, page in enumerate(pdf_pages):
                    image_array = np.array(page)
                    images.append((f"Page {i+1}", image_array, page))
            else:
                image_array, display_image = process_uploaded_file(uploaded_file)
                images = [("Page 1", image_array, display_image)]
            
            page_names = [name for name, _, _ in images]
            selected_page = st.selectbox("📄 Select PDF Page", page_names)

            # Lấy trang được chọn
            for name, image_array, display_image in images:
                if name == selected_page:
                    current_array = image_array
                    current_display = display_image
                    break
            
            if current_array is None or current_display is None:
                return
                        
            # Hiển thị ảnh + ROI
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("### 🖼️ Document for Processing")
                st.image(current_display, use_column_width=True)
                st.info(f"📐 Image size: {current_array.shape[1]}×{current_array.shape[0]} pixels")
            
            with col2:
                st.markdown("### 🎯 ROI Detection Map")
                if st.session_state.ocr_service:
                    roi_fig = create_roi_visualization(current_array.shape)
                    if roi_fig:
                        st.plotly_chart(roi_fig, use_container_width=True)
                    else:
                        st.info("Unable to create ROI visualization")
                else:
                    st.info("Initialize OCR service to see ROI map")
            
            # Processing section
            st.markdown("### ⚙️ Processing Control")
            
            if st.button("🚀 Start ROI-OCR Processing", type="primary", use_container_width=True):
                ocr_service = load_ocr_service()
                if ocr_service is None:
                    st.error("Failed to initialize OCR service. Please check your dependencies.")
                    return
                
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    start_time = time.time()
                    status_text.text("🔍 Extracting ROI regions and preprocessing...")
                    progress_bar.progress(20)
                    
                    status_text.text("🤖 Running hierarchical OCR on each field...")
                    progress_bar.progress(50)
                    
                    results = ocr_service.hierarchical_ocr(current_array)
                    progress_bar.progress(80)
                    
                    status_text.text("✅ Validating extracted fields...")
                    progress_bar.progress(100)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    status_text.success(f"✅ ROI-OCR processing complete! ({processing_time:.2f}s)")
                    
                    st.session_state.results = results
                    st.session_state.processing_time = processing_time
                    
                except Exception as e:
                    st.error(f"❌ Error during OCR processing: {str(e)}")
                    logger.error(f"OCR processing error: {e}")
                    return
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            logger.error(f"Main file processing error: {e}")
            return
    
    # Display results if available
    if st.session_state.results is not None:
        st.markdown("---")
        st.markdown("## 📈 Processing Results")
        
        # Summary metrics
        create_summary_metrics(st.session_state.results)
        
        # Processing time and method info
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "⏱️ Processing Time", 
                f"{st.session_state.processing_time:.2f}s",
                help="Total time for ROI extraction and OCR processing"
            )
        with col2:
            st.metric(
                "🔧 Processing Method", 
                st.session_state.results['processing_method'],
                help="OCR processing approach used"
            )
        
        # Confidence chart
        confidence_fig = create_confidence_chart(st.session_state.results)
        if confidence_fig:
            st.plotly_chart(confidence_fig, use_container_width=True)
        
        # Field results
        display_field_results(st.session_state.results)
        
        # Export options
        st.markdown("### 📤 Export Options")
        
        try:
            # Add processing info to export data
            processing_info = {
                'file_name': uploaded_file.name if uploaded_file else 'unknown',
                'file_type': uploaded_file.type if uploaded_file else 'unknown',
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export field data as JSON
                export_data = {
                    'document_info': processing_info,
                    'extracted_fields': {
                        field: {
                            'text': data.get('final_text', data['text']),
                            'confidence': data['confidence'],
                            'method': data['method_used'],
                            'validated': data.get('validation', {}).get('is_valid', False)
                        }
                        for field, data in st.session_state.results['processed_fields'].items()
                        if data['text']
                    },
                    'processing_summary': st.session_state.results['summary'],
                    'processing_time': st.session_state.processing_time
                }
                
                json_str = json.dumps(export_data, indent=2)
                
                filename = f"insurance_form_data.json"
                st.download_button(
                    label="📋 Download Field Data (JSON)",
                    data=json_str,
                    file_name=filename,
                    mime="application/json"
                )
            
            with col2:
                # Export as structured CSV
                csv_data = []
                for field, data in st.session_state.results['processed_fields'].items():
                    if data['text']:
                        csv_data.append({
                            'Field': field.replace('_', ' ').title(),
                            'Extracted_Text': data.get('final_text', data['text']),
                            'Confidence': f"{data['confidence']:.1%}",
                            'OCR_Method': data['method_used'],
                            'Validated': data.get('validation', {}).get('is_valid', False)
                        })
                
                if csv_data:
                    df = pd.DataFrame(csv_data)
                    csv = df.to_csv(index=False)
                    
                    filename = f"insurance_form_fields.csv"
                    st.download_button(
                        label="📊 Download as CSV",
                        data=csv,
                        file_name=filename,
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error creating export options: {e}")
            logger.error(f"Export options error: {e}")
        
        # Detailed analysis
        try:
            with st.expander("🔍 Detailed Field Analysis", expanded=False):
                for field_name, field_data in st.session_state.results['processed_fields'].items():
                    st.markdown(f"**{field_name.replace('_', ' ').title()}**")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"ROI Size: {field_data['roi_size']}")
                    with col2:
                        st.write(f"Final Method: {field_data['method_used']}")
                    with col3:
                        st.write(f"Confidence: {field_data['confidence']:.1%}")
                    
                    if 'attempts' in field_data:
                        attempts_df = pd.DataFrame([
                            {'Method': method.upper(), 'Text': attempt['text'][:50] + '...' if len(attempt['text']) > 50 else attempt['text'], 'Confidence': f"{attempt['confidence']:.1%}"}
                            for method, attempt in field_data['attempts'].items()
                        ])
                        st.dataframe(attempts_df, use_container_width=True)
                    
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error in detailed analysis: {e}")
            logger.error(f"Detailed analysis error: {e}")

if __name__ == "__main__":
    main()