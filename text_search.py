import streamlit as st
import numpy as np
from image_utils import cosine_similarity_np
from uform import get_model, Modality
import torch
from PIL import Image

# Load model và processor
processors, models = get_model('unum-cloud/uform3-image-text-english-small')
model_text = models[Modality.TEXT_ENCODER]
model_image = models[Modality.IMAGE_ENCODER]
processor_text = processors[Modality.TEXT_ENCODER]
processor_image = processors[Modality.IMAGE_ENCODER]

def text_search(src_imgs):
    st.subheader("Search by Text")
    query_text = st.text_input("Enter a description (English)")
    if not query_text:
        st.info("Please enter some text to search.")
        return

    min_similar = st.slider(
                    'Minimum Similarity (%)',
                    value=70, min_value=10, max_value=99, step=5,
                    key="text_similarity"
                )


    with st.spinner("Encoding text..."):
        text_data = processor_text(query_text)
        _, text_emb = model_text.encode(text_data, return_features=True)

    results = []
    with st.spinner("Searching images..."):
        for img in src_imgs:
            image_data = processor_image(img)
            _, img_emb = model_image.encode(image_data, return_features=True)
            sim = cosine_similarity_np(img_emb.flatten(), text_emb.flatten()) * 100
            if sim >= min_similar:
                results.append((sim, img))

    if not results:
        st.info("No results found.")
        return

    results.sort(reverse=True, key=lambda x: x[0])
    st.success(f"Found {len(results)} results")

    cols = st.columns(4)
    for idx, (score, matched_img) in enumerate(results):
        with cols[idx % 4]:
            st.image(
                matched_img,
                caption=f"{round(score)}%",
                use_container_width=True,
            )
