import numpy as np
import streamlit as st
from image_utils import cosine_similarity_np, get_image_embeddings,process_image
import cv2

def face_search(src_imgs):
    st.subheader("Search by Face")
    option = st.radio("Choose input method:", ("Take a picture", "Upload a photo"))
    query_embs = None
    query_img = None
    results = []
    
    if option == "Take a picture":
        img_file = st.camera_input('📷 Take a picture')
        if img_file is not None:
            bytes_data = img_file.getvalue()
            query_img = process_image(bytes_data)
    else:
        uploaded_img = st.file_uploader("Upload a photo", type=['jpg', 'jpeg', 'png'])
        if uploaded_img is not None:
            bytes_data = uploaded_img.read()
            query_img = process_image(bytes_data)

    if query_img is not None:
        with st.spinner("Extracting query face embedding..."):
            query_embs = get_image_embeddings(query_img)

        if query_embs is None:
            st.warning('Face not found', icon='⚠️')
            return

        min_similar = st.slider(
                        'Minimum Similarity (%)',
                        value=70, min_value=10, max_value=99, step=5,
                        key="face_similarity"
                    )
                
        with st.spinner("Searching database..."):            
            for i, img in enumerate(src_imgs):
                src_img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                src_embs = get_image_embeddings(src_img_cv)

                if src_embs is not None:
                    sim = cosine_similarity_np(src_embs, query_embs) * 100
                    if sim >= min_similar:
                        results.append((sim, img))
                            
        if len(results) == 0:
                st.info("No similar images found.")
        else:
            results.sort(reverse=True, key=lambda x: x[0])

            st.success(f"Found {len(results)} similar images")
            cols = st.columns(4)
            for j, (score, matched_img) in enumerate(results):
                with cols[j % 4]:
                    st.image(matched_img, caption=f"{round(score)}%", use_container_width=True)
    else:
        st.info("Please take a picture or upload a photo.")