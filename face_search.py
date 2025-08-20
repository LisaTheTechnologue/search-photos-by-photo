import numpy as np
import streamlit as st
from image_utils import get_image_embeddings
import cv2

def face_search(src_imgs):
    st.subheader("Search by Face")
    option = st.radio("Choose input method:", ("Take a picture", "Upload a photo"))
    embs = None
    img = None
    results = []
    if option == "Take a picture":
        img_file = st.camera_input('📷 Take a picture')
        if img_file is not None:
            bytes_data = img_file.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    else:
        uploaded_img = st.file_uploader("Upload a photo", type=['jpg', 'jpeg', 'png'])
        if uploaded_img is not None:
            bytes_data = uploaded_img.read()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if img is not None:
        with st.spinner("Extracting query face embedding..."):
            embs = get_image_embeddings(img)

        if embs is None:
            st.warning('Face not found', icon='⚠️')
            return

        min_similar = st.slider('Minimum Similarity', value=70, min_value=10, max_value=99, step=5)
        
        with st.spinner("Searching database..."):            
            for i, img in enumerate(src_imgs):
                src_embs = get_image_embeddings(np.array(img))
                # Tính cosine sim giữa các gương mặt và kết quả
                if src_embs is not None:
                    cosine = (src_embs @ embs.T).flatten() * 100
                    for score in cosine:
                        if score >= min_similar:
                            results.append((score, img))
                            
        if len(results) == 0:
                st.info("No similar images found.")
        else:
            # sort high → low
            results.sort(reverse=True, key=lambda x: x[0])

            st.success(f"Found {len(results)} similar images")
            cols = st.columns(4)
            for j, (score, matched_img) in enumerate(results):
                with cols[j % 4]:
                    st.image(matched_img, caption=f"{round(score)}%", use_container_width=True)
    else:
        st.info("Please take a picture or upload a photo.")