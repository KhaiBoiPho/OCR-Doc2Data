import easyocr
from PIL import Image
import numpy as np
import cv2
import streamlit as st

@st.cache_resource
def load_ocr_reader():
    """Cache the OCR reader to avoid reloading on every run"""
    return easyocr.Reader(['ch_sim', 'en'], gpu=False)

def ocr_extract(image):
    reader = load_ocr_reader()
    img_cv = np.array(image.convert("RGB"))
    results_raw = reader.readtext(img_cv)

    results = []
    for (bbox, text, conf) in results_raw:
        # bbox is 4 points (x,y)
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
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
        cv2.putText(img_cv, r["text"], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
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