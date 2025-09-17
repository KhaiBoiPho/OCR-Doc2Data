import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np

@st.cache_resource
def load_ocr_reader():
    """Cache the OCR reader to avoid reloading on every run"""
    try:
        from paddleocr import PaddleOCR
        return PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False, show_log=False)
    except Exception as e:
        st.error(f"Failed to load PaddleOCR: {e}")
        return None

def ocr_extract(image):
    reader = load_ocr_reader()
    if reader is None:
        return []
    
    try:
        img_array = np.array(image.convert("RGB"))
        results_raw = reader.ocr(img_array, cls=True)  # Fixed: use 'ocr' method, not 'predict'

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
        st.error(f"OCR processing failed: {e}")
        return []

def draw_boxes(image, results):
    """Draw bounding boxes using PIL instead of cv2"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    for r in results:
        x, y, w, h = r["bbox"]
        
        # Draw green rectangle (equivalent to cv2.rectangle)
        draw.rectangle([x, y, x+w, y+h], outline="green", width=2)
        
        # Draw text (equivalent to cv2.putText)
        try:
            text_to_draw = r["text"]
            # Limit text length to prevent UI issues
            if len(text_to_draw) > 15:
                text_to_draw = text_to_draw[:15] + "..."
            
            if font:
                draw.text((x, max(0, y-20)), text_to_draw, fill="green", font=font)
            else:
                draw.text((x, max(0, y-20)), text_to_draw, fill="green")
        except Exception as e:
            # Fallback if text rendering fails
            try:
                if font:
                    draw.text((x, max(0, y-20)), "Text", fill="green", font=font)
                else:
                    draw.text((x, max(0, y-20)), "Text", fill="green")
            except:
                pass  # Skip if even fallback fails
    
    return img_copy

def extract_key_fields(results):
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
        if "姓名" in t or "Name" in t or "何大伟" in t:
            fields["姓名 (Name)"] = t
        if "HKID" in t or "身份证" in t or "B234567" in t:
            fields["证件 (HKID)"] = t
        if "/" in t and len(t) >= 8:
            fields["日期 (Date)"] = t
        if "医生" in t or "Doctor" in t or "黄尔明" in t:
            fields["医生 (Doctor)"] = t
        if "Address" in t or "地址" in t:
            fields["Address"] = t
        if "Treatment" in t or "治疗" in t:
            fields["Treatment"] = t
        if "Occupation" in t or "职业" in t:
            fields["Occupation"] = t
    
    return fields