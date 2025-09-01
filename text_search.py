import streamlit as st
import numpy as np
from image_utils import cosine_similarity_np
from uform import get_model, Modality
# import torch
import io
from PIL import Image

# Load UForm model & processor
processors, models = get_model("unum-cloud/uform3-image-text-english-small")
model_text = models[Modality.TEXT_ENCODER]
model_image = models[Modality.IMAGE_ENCODER]
processor_text = processors[Modality.TEXT_ENCODER]
processor_image = processors[Modality.IMAGE_ENCODER]

# device = "cuda" if torch.cuda.is_available() else "cpu"
# try:
#     model_text = model_text.to(device)
#     model_image = model_image.to(device)
# except AttributeError:
#     pass  # ONNX backend không có .to()


# Cache embeddings ảnh cho Text Search
@st.cache_data
def compute_image_embeddings_text(imgs_bytes):
    embs = []
    for bytes_data in imgs_bytes:
        img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        image_data = processor_image(img)
        _, img_emb = model_image.encode(image_data, return_features=True)
        # if hasattr(img_emb, "detach"):  # nếu là torch.Tensor
        #     img_emb = img_emb.detach().cpu().numpy()
        embs.append(img_emb)
    return embs


def text_search(src_imgs):
    st.subheader("Search by Text")

    query_text = st.text_input("Enter a description (English)")
    if not query_text:
        st.info("Please enter some text to search.")
        return

    min_similar = st.slider(
        "Minimum Similarity (%)",
        value=70, min_value=10, max_value=99, step=5,
        key="text_slider"
    )

    # Convert src_imgs to bytes
    imgs_bytes = []
    for img in src_imgs:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        imgs_bytes.append(buf.getvalue())

    # Lấy embeddings từ cache
    src_embs = compute_image_embeddings_text(imgs_bytes)

    with st.spinner("Encoding text..."):
        text_data = processor_text(query_text)
        _, text_emb = model_text.encode(text_data, return_features=True)
        # if hasattr(text_emb, "detach"):
        #     text_emb = text_emb.detach().cpu().numpy()

    results = []
    with st.spinner("Searching images..."):
        for emb, img in zip(src_embs, src_imgs):
            if emb is not None:
                sim = cosine_similarity_np(emb.flatten(), text_emb.flatten()) * 100
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
