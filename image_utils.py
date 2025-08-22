from PIL import Image, ImageOps
from zipfile import ZipFile
from deepface import DeepFace
import numpy as np
import streamlit as st
import cv2

def cosine_similarity_np(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_image_embeddings(img):
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Hàm represent của thư viện deepface 
        # để biểu diễn gương mặt cần tìm 
        # và các gương mặt có trong file zip
        embs = DeepFace.represent(
            img_rgb,
            model_name="Facenet",
            enforce_detection=True  
        )
        return np.array(embs[0]['embedding'])
    except:
        return None

def process_image(bytes_data):
    return cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)


def read_zip_file(uploaded_file):
    imgs = []
    with ZipFile(uploaded_file, 'r') as zip:
        image_files = [f for f in zip.namelist() if f.lower().endswith(('png', 'jpg', 'jpeg'))]

        if image_files:
            cols = st.columns(4)  
            for i, filename in enumerate(image_files):
                with zip.open(filename) as f:
                    img = Image.open(f).convert("RGB")
                    img = ImageOps.exif_transpose(img)
                    imgs.append(img)
                    with cols[i % 4]:
                        st.image(img, caption=filename[-20:], use_container_width=True)
    return imgs