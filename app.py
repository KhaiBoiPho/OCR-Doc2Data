import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
import backend

st.set_page_config(page_title="Insurance OCR Demo", layout="wide")

st.title("📄 Insurance Claim Form OCR Demo")
st.write("Upload an insurance claim form (image/PDF) and extract key fields.")

uploaded_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    images = []
    if uploaded_file.type == "application/pdf":
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in pdf:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    else:
        images.append(Image.open(uploaded_file))

    for i, image in enumerate(images):
        st.image(image, caption=f"Page {i+1} - Uploaded Document", use_column_width=True)

        with st.spinner("Processing OCR..."):
            try:
                results = backend.ocr_extract(image)
                boxed_image = backend.draw_boxes(image, results)
                fields = backend.extract_key_fields(results)

                st.subheader(f"🖼 OCR with Bounding Boxes - Page {i+1}")
                st.image(boxed_image, use_column_width=True)

                st.subheader(f"📑 Extracted Fields - Page {i+1}")
                st.json(fields)
            except Exception as e:
                st.error(f"OCR processing failed: {str(e)}")
                st.info("Please check your dependencies and try again.")