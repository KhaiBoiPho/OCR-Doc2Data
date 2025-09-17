import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np

@st.cache_resource
def load_ocr_reader():
    """Cache the OCR reader to avoid reloading on every run"""
    try:
        from paddleocr import PaddleOCR
        return PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False, show_log=False)
    except ImportError:
        st.error("PaddleOCR not available. Using demo mode.")
        return None

def ocr_extract(image):
    """Extract text from image using PaddleOCR or demo data"""
    reader = load_ocr_reader()
    
    if reader is None:
        # Fallback to demo data if PaddleOCR fails
        return get_demo_results(image.size)
    
    try:
        img_array = np.array(image.convert("RGB"))
        results_raw = reader.ocr(img_array, cls=True)

        results = []
        if results_raw and results_raw[0]:  # PaddleOCR returns nested structure
            for line in results_raw[0]:
                if len(line) >= 2:
                    bbox_points, (text, conf) = line
                    # Convert bbox points to rectangle
                    xs = [pt[0] for pt in bbox_points]
                    ys = [pt[1] for pt in bbox_points]
                    x_min, y_min = int(min(xs)), int(min(ys))
                    x_max, y_max = int(max(xs)), int(max(ys))
                    
                    results.append({
                        "text": text,
                        "conf": float(conf),
                        "bbox": (x_min, y_min, x_max - x_min, y_max - y_min)
                    })
        return results
    except Exception as e:
        st.warning(f"OCR failed, using demo mode: {e}")
        return get_demo_results(image.size)

def draw_boxes(image, results):
    """Draw bounding boxes on image without cv2"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to use a font
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for r in results:
        x, y, w, h = r["bbox"]
        
        # Draw green rectangle
        draw.rectangle([x, y, x+w, y+h], outline="green", width=2)
        
        # Draw text above the box
        try:
            # Limit text length to prevent overflow
            display_text = r["text"][:20] + "..." if len(r["text"]) > 20 else r["text"]
            draw.text((x, max(0, y-15)), display_text, fill="green", font=font)
        except:
            # Fallback for problematic text
            draw.text((x, max(0, y-15)), "Text", fill="green", font=font)
    
    return img_copy

def get_demo_results(image_size):
    """Generate demo OCR results"""
    w, h = image_size
    
    demo_results = [
        {"text": "保险理赔表格", "conf": 0.95, "bbox": (w//4, 50, w//2, 30)},
        {"text": "Insurance Claim Form", "conf": 0.93, "bbox": (w//4, 90, w//2, 25)},
        {"text": "姓名: 何大伟", "conf": 0.90, "bbox": (50, 150, 120, 25)},
        {"text": "Name: John Doe", "conf": 0.88, "bbox": (200, 150, 100, 25)},
        {"text": "HKID: B234567(8)", "conf": 0.92, "bbox": (50, 200, 130, 25)},
        {"text": "日期: 15/09/2024", "conf": 0.89, "bbox": (50, 250, 120, 25)},
        {"text": "Date: 15/09/2024", "conf": 0.87, "bbox": (200, 250, 100, 25)},
        {"text": "医生: 黄尔明", "conf": 0.85, "bbox": (50, 300, 100, 25)},
        {"text": "Doctor: Dr. Wong", "conf": 0.83, "bbox": (200, 300, 120, 25)},
        {"text": "Address: Hong Kong", "conf": 0.81, "bbox": (50, 350, 150, 25)},
        {"text": "Treatment: Consultation", "conf": 0.79, "bbox": (50, 400, 180, 25)},
        {"text": "Occupation: Engineer", "conf": 0.77, "bbox": (50, 450, 160, 25)},
    ]
    
    return demo_results

def extract_key_fields(results):
    """Extract key insurance form fields"""
    fields = {
        "姓名 (Name)": "", 
        "证件 (HKID)": "", 
        "出生日期 (DOB)": "", 
        "医生 (Doctor)": "", 
        "日期 (Date)": "",
        "Address": "",
        "Treatment": "",
        "Occupation": ""
    }
    
    for r in results:
        t = r["text"]
        
        # Match names
        if "姓名" in t or "Name" in t or "何大伟" in t or "John Doe" in t:
            fields["姓名 (Name)"] = t
            
        # Match ID numbers  
        elif "HKID" in t or "身份证" in t or "B234567" in t:
            fields["证件 (HKID)"] = t
            
        # Match dates
        elif "/" in t and any(char.isdigit() for char in t) and len(t) >= 8:
            fields["日期 (Date)"] = t
            
        # Match doctors
        elif "医生" in t or "Doctor" in t or "黄尔明" in t or "Wong" in t:
            fields["医生 (Doctor)"] = t
            
        # Match addresses
        elif "Address" in t or "地址" in t or "Hong Kong" in t:
            fields["Address"] = t
            
        # Match treatments
        elif "Treatment" in t or "治疗" in t or "Consultation" in t:
            fields["Treatment"] = t
            
        # Match occupations
        elif "Occupation" in t or "职业" in t or "Engineer" in t:
            fields["Occupation"] = t
    
    return fields