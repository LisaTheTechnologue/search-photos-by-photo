import streamlit as st
import numpy as np
from numpy.linalg import norm
from image_utils import get_image_embeddings
from uform import get_model

model, processor = get_model("unum-cloud/uform-vl-english-small",backend="torch")

def get_text_embedding(text: str):
    text_features, text_emb = model.encode_text([text], return_features=True)
    text_emb = text_emb[0] / np.linalg.norm(text_emb[0])  # normalize
    return text_emb

def text_search(src_imgs):
    # col1, col2 = st.columns(2)
    # with col1:
    text = st.text_area('Image Description')
    # with col2:
    min_similar = st.slider('Minimum Similarity', value=70, min_value=10, max_value=99, step=5)
    embs = get_text_embedding(text)
    results = []

    if len(text) > 0:
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
        # Lấy embeddings ảnh
        # src_embs = [get_image_embeddings(np.array(i)) for i in src_imgs]
        # src_embs = [e for e in src_embs if e is not None]
        # src_embs = np.vstack(src_embs)

        # # Lấy embedding text
        # text_embedding = get_text_embedding(text)

        # # Cosine similarity
        # cosine = (src_embs @ text_embedding) * 100
        # ids = np.where(cosine >= min_cosine)[0]

        # if len(ids) == 0:
        #     st.info('Not found')
        # else:
        #     result = [(cosine[i], i) for i in ids]
        #     result.sort(reverse=True)

        #     st.success('Result')
        #     cols = st.columns(3)
        #     for j, (cos_val, i) in enumerate(result):
        #         with cols[j % 3]:
        #             st.image(src_imgs[i], f'{round(cos_val)}%')