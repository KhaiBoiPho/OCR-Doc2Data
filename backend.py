from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2

ocr = PaddleOCR(use_angle_cls=True, lang="ch")  

def ocr_extract(image):
    img_cv = np.array(image.convert("RGB"))
    result = ocr.ocr(img_cv, cls=True)

    results = []
    if result and len(result[0]) > 0:
        for line in result[0]:
            bbox = line[0]
            try:
                xs = [float(pt[0]) for pt in bbox]
                ys = [float(pt[1]) for pt in bbox]
            except:
                continue

            if isinstance(line[1], (list, tuple)) and len(line[1]) == 2:
                text, conf = line[1]
            else:
                text, conf = str(line[1]), 1.0

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
    return Image.fromarray(img_cv)

def extract_key_fields(results):
    fields = {"姓名 (Name)": "", "证件 (HKID)": "", "出生日期 (DOB)": "", "医生 (Doctor)": "", "日期 (Date)": ""}
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
        if "Adderess" in t:
            fields["Address"] = t
        if "Treatment" in t:
            fields["Treatment"] = t
        if "Occupation" in t:
            fields["Occupation"] = t
    return fields