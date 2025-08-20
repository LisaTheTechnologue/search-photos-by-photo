from PIL import Image, ImageOps
from zipfile import ZipFile
from deepface import DeepFace
import numpy as np
import streamlit as st
import cv2

def normalize(a):
    # A_norm = A / norm(A, axis=1).reshape(-1,1)
    n = np.norm(a, axis=1)
    return a / n.reshape(-1, 1)

def get_image_embeddings(img):
    try:
        cv2.imwrite("temp.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # Hàm represent của thư viện deepface 
        # để biểu diễn gương mặt cần tìm 
        # và các gương mặt có trong file zip
        embs = DeepFace.represent("temp.jpg", model_name="Facenet")
        return normalize(np.array([e['embedding'] for e in embs]))
    except:
        return None

def read_zip_file(uploaded_file):
    imgs = []
    with ZipFile(uploaded_file, 'r') as zip:
        image_files = [f for f in zip.namelist() if f.lower().endswith(('png', 'jpg', 'jpeg'))]

        if image_files:
            cols = st.columns(4)  # 4 images per row
            for i, filename in enumerate(image_files):
                with zip.open(filename) as f:
                    img = Image.open(f).convert("RGB")
                    img = ImageOps.exif_transpose(img)
                    imgs.append(img)
                    # Show each image in its column
                    with cols[i % 4]:
                        st.image(img, caption=filename[-20:], use_container_width=True)
    return imgs