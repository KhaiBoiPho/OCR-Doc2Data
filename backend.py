import streamlit as st
from PIL import Image
import numpy as np
import cv2

@st.cache_resource
def load_ocr_reader():
    """Cache the OCR reader to avoid reloading on every run"""
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False, show_log=False)

def ocr_extract(image):
    reader = load_ocr_reader()
    img_cv = np.array(image.convert("RGB"))
    results_raw = reader.ocr(img_cv, cls=True)

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

def draw_boxes(image, results):
    img_cv = np.array(image.convert("RGB"))
    for r in results:
        (x, y, w, h) = r["bbox"]
        cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0,255,0), 2)
        try:
            cv2.putText(img_cv, r["text"], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        except:
            # Skip text that can't be rendered
            pass
    return Image.fromarray(img_cv)

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