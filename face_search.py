import numpy as np
import streamlit as st
from image_utils import cosine_similarity_np, get_image_embeddings,process_image
import cv2
from PIL import Image
import io

@st.cache_data
def compute_face_embeddings(imgs_bytes):
    embs = []
    for bytes_data in imgs_bytes:
        img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        emb = get_image_embeddings(img_cv)
        embs.append(emb)
    return embs

def face_search(src_imgs):
    st.subheader("Search by Face")
    option = st.radio(
        "Choose input method:", ("Take a picture", "Upload a photo"), key="face_input"
    )

    query_img = None
    if option == "Take a picture":
        img_file = st.camera_input("📷 Take a picture", key="face_camera")
        if img_file is not None:
            query_img = process_image(img_file.getvalue())
    else:
        uploaded_img = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="face_upload")
        if uploaded_img is not None:
            query_img = process_image(uploaded_img.read())

    if query_img is None:
        st.info("Please take a picture or upload a photo.")
        return

    with st.spinner("Extracting query face embedding..."):
        query_emb = get_image_embeddings(query_img)
        if query_emb is None:
            st.warning("Face not found", icon="⚠️")
            return

    min_similar = st.slider(
        "Minimum Similarity (%)",
        value=70, min_value=10, max_value=99, step=5,
        key="face_slider"
    )

    # Convert src_imgs to bytes
    imgs_bytes = []
    for img in src_imgs:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        imgs_bytes.append(buf.getvalue())

    with st.spinner("Searching database..."):
        src_embs = compute_face_embeddings(imgs_bytes)
        results = []
        for emb, img in zip(src_embs, src_imgs):
            if emb is not None:
                sim = cosine_similarity_np(emb, query_emb) * 100
                if sim >= min_similar:
                    results.append((sim, img))

    if not results:
        st.info("No similar images found.")
    else:
        results.sort(reverse=True, key=lambda x: x[0])
        st.success(f"Found {len(results)} similar images")

        cols = st.columns(4)
        for idx, (score, matched_img) in enumerate(results):
            with cols[idx % 4]:
                st.image(matched_img, caption=f"{round(score)}%", use_container_width=True)