import pytesseract
from pytesseract import Output
import cv2
from PIL import Image
import numpy as np

def ocr_extract(image):
    """
    Chạy OCR bằng Tesseract và trả về bounding boxes + text.
    """
    img_cv = np.array(image.convert("RGB"))
    d = pytesseract.image_to_data(img_cv, output_type=Output.DICT)

    results = []
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 50:  # confidence > 50%
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            results.append({
                "text": d['text'][i],
                "conf": d['conf'][i],
                "bbox": (x, y, w, h)
            })
    return results

def draw_boxes(image, results):
    """
    Vẽ bounding box lên ảnh OCR kết quả.
    """
    img_cv = np.array(image.convert("RGB"))
    for r in results:
        (x, y, w, h) = r["bbox"]
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return Image.fromarray(img_cv)

def extract_key_fields(results):
    """
    Trích xuất 1 số trường chính từ OCR.
    Ở đây làm rule-based đơn giản, có thể cải tiến bằng regex/LLM.
    """
    fields = {
        "Name": "",
        "HKID": "",
        "DOB": "",
        "Doctor": "",
        "Date": ""
    }

    text = " ".join([r["text"] for r in results])

    # Rule-based search
    for r in results:
        t = r["text"]
        if "何大伟" in t or "Name" in t:
            fields["Name"] = t
        if "B234567(8)" in t or "HKID" in t:
            fields["HKID"] = t
        if "06/02/2022" in t or "/" in t:
            fields["Date"] = t
        if "黄尔明" in t or "Doctor" in t:
            fields["Doctor"] = t
        if "Adderess" in t:
            fields["Address"] = t
        if "Treatment" in t:
            fields["Treatment"] = t
        if "Occupation" in t:
            fields["Occupation"] = t

    return fields
