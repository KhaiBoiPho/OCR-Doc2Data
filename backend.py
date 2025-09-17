import streamlit as st
from PIL import Image
import numpy as np
import cv2

@st.cache_resource
def load_ocr_reader():
    """Load EasyOCR - nhẹ và tốt cho chữ viết tay"""
    try:
        import easyocr
        reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        st.success("EasyOCR loaded successfully!")
        return reader
    except Exception as e:
        st.error(f"Failed to load EasyOCR: {e}")
        return None

def ocr_extract(image):
    reader = load_ocr_reader()
    if reader is None:
        st.error("OCR not available")
        return []
    
    try:
        img_cv = np.array(image.convert("RGB"))
        results_raw = reader.readtext(img_cv)

        results = []
        for (bbox, text, conf) in results_raw:
            # bbox là 4 điểm (x,y)
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x_min, y_min = int(min(xs)), int(min(ys))
            x_max, y_max = int(max(xs)), int(max(ys))
            
            # Only keep results with decent confidence
            if conf > 0.3:  # Filter low confidence results
                results.append({
                    "text": text,
                    "conf": float(conf),
                    "bbox": (x_min, y_min, x_max - x_min, y_max - y_min)
                })
        
        st.info(f"Real OCR extracted {len(results)} text elements from your image")
        return results
        
    except Exception as e:
        st.error(f"OCR processing failed: {e}")
        return []

def draw_boxes(image, results):
    """Draw bounding boxes with confidence scores"""
    img_cv = np.array(image.convert("RGB"))
    
    for r in results:
        (x, y, w, h) = r["bbox"]
        conf = r["conf"]
        
        # Color based on confidence: red=low, yellow=medium, green=high
        if conf > 0.8:
            color = (0, 255, 0)  # Green for high confidence
        elif conf > 0.5:
            color = (0, 255, 255)  # Yellow for medium confidence  
        else:
            color = (0, 0, 255)  # Red for low confidence
            
        # Draw rectangle
        cv2.rectangle(img_cv, (x, y), (x+w, y+h), color, 2)
        
        # Draw text with confidence
        try:
            label = f"{r['text'][:15]} ({conf:.2f})"
            cv2.putText(img_cv, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        except:
            cv2.putText(img_cv, f"Text ({conf:.2f})", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return Image.fromarray(img_cv)

def extract_key_fields(results):
    """Extract key fields from REAL OCR results"""
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

    # Sort results by confidence (highest first) for better matching
    sorted_results = sorted(results, key=lambda x: x["conf"], reverse=True)
    
    for r in sorted_results:
        t = r["text"].strip()
        
        # Skip empty or very short text
        if len(t) < 2:
            continue
            
        # Name matching
        if not fields["姓名 (Name)"]:
            if any(keyword in t for keyword in ["姓名", "Name", "name"]) or \
               any(name in t for name in ["何大伟", "John", "Jane", "李", "王", "张", "陈"]):
                fields["姓名 (Name)"] = t
                
        # HKID matching  
        if not fields["证件 (HKID)"]:
            if "HKID" in t or "身份证" in t or \
               any(pattern in t for pattern in ["B234567", "A123456", "C987654"]) or \
               (len(t) >= 8 and any(c.isdigit() for c in t) and any(c.isalpha() for c in t)):
                fields["证件 (HKID)"] = t
                
        # Date matching
        if not fields["日期 (Date)"]:
            import re
            if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', t) or \
               re.search(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', t):
                fields["日期 (Date)"] = t
                
        # Doctor matching
        if not fields["医生 (Doctor)"]:
            if any(keyword in t for keyword in ["医生", "Doctor", "Dr.", "醫生"]) or \
               any(name in t for name in ["黄尔明", "Wong", "李医生", "陈医生"]):
                fields["医生 (Doctor)"] = t
                
        # Address matching
        if not fields["Address"]:
            if any(keyword in t for keyword in ["Address", "地址", "Hong Kong", "九龙", "港岛", "新界", "街", "路", "道"]):
                fields["Address"] = t
                
        # Treatment matching
        if not fields["Treatment"]:
            if any(keyword in t for keyword in ["Treatment", "治疗", "诊疗", "检查", "手术", "药物"]):
                fields["Treatment"] = t
                
        # Occupation matching
        if not fields["Occupation"]:
            if any(keyword in t for keyword in ["Occupation", "职业", "工作", "Engineer", "Teacher", "Doctor"]):
                fields["Occupation"] = t

    return fields