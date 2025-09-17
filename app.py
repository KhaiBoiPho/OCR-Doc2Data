import streamlit as st
from PIL import Image
import backend
import fitz  # PyMuPDF

st.set_page_config(page_title="Insurance OCR Demo", layout="wide")

st.title("📄 Insurance Claim Form OCR Demo")
st.write("Upload an insurance claim form (image/PDF) and extract key fields.")

uploaded_file = st.file_uploader("Upload Document", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    images = []

    if uploaded_file.type == "application/pdf":
        # Convert PDF pages -> images
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap()  
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    else:
        # Regular image
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