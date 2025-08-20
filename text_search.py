import streamlit as st
import numpy as np
from numpy.linalg import norm
from image_utils import get_image_embeddings
from uform import get_model

model, tokenizer, processor = get_model("unum-cloud/uform-vl-multilingual", backend="torch")

def get_text_embedding(text: str):
    text_emb = model.encode_text([text], tokenizer)
    text_emb = text_emb[0] / np.linalg.norm(text_emb[0])  # normalize
    return text_emb

def text_search(src_imgs):
    col1, col2 = st.columns(2)
    with col1:
        text = st.text_area('Image Description')
    with col2:
        min_cosine = st.slider('Level of Similarity (%)', value=30, min_value=10, max_value=100, step=5)

    if len(text) > 0:
        # Lấy embeddings ảnh
        src_embs = [get_image_embeddings(np.array(i)) for i in src_imgs]
        src_embs = [e for e in src_embs if e is not None]
        src_embs = np.vstack(src_embs)

        # Lấy embedding text
        text_embedding = get_text_embedding(text)

        # Cosine similarity
        cosine = (src_embs @ text_embedding) * 100
        ids = np.where(cosine >= min_cosine)[0]

        if len(ids) == 0:
            st.info('Not found')
        else:
            result = [(cosine[i], i) for i in ids]
            result.sort(reverse=True)

            st.success('Result')
            cols = st.columns(3)
            for j, (cos_val, i) in enumerate(result):
                with cols[j % 3]:
                    st.image(src_imgs[i], f'{round(cos_val)}%')