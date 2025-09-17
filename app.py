import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
import backend

st.set_page_config(page_title="Insurance OCR Demo", layout="wide")

st.title("📄 Insurance Claim Form OCR Demo")
st.write("Upload an insurance claim form (image/PDF) and extract key fields.")

uploaded_file = st.file_uploader("Upload Document", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    images = []

    if uploaded_file.type == "application/pdf":
        # PDF -> list of PIL images (mỗi trang là 1 ảnh)
        images = convert_from_bytes(uploaded_file.read())
    else:
        # Ảnh thường
        images = [Image.open(uploaded_file)]

    for page_num, image in enumerate(images, start=1):
        st.subheader(f"📄 Page {page_num}")
        st.image(image, caption=f"Uploaded Page {page_num}", use_column_width=True)

        with st.spinner("Processing OCR..."):
            results = backend.ocr_extract(image)
            boxed_image = backend.draw_boxes(image, results)
            fields = backend.extract_key_fields(results)

        st.subheader("🖼 OCR with Bounding Boxes")
        st.image(boxed_image, use_column_width=True)

        st.subheader("📑 Extracted Fields")
        st.json(fields)