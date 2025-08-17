import streamlit as st
import numpy as np
import pandas as pd
from zipfile import ZipFile
from PIL import Image
import io
from numpy.linalg import norm
from deepface import DeepFace
import cv2

@st.cache_data
def normalize(a):
    n = norm(a, axis=1)
    return a / n.reshape(-1, 1)

@st.cache_data
def get_image_embeddings(img):
    try:
        embs = DeepFace.represent(img)
        return normalize(np.array([e['embedding'] for e in embs]))
    except:
        return None

def face_search(src_imgs):
    embs = None
    img_file = st.camera_input('📷 Take a picture')
    
    if img_file is not None:
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        embs = get_image_embeddings(img)

        if embs is None:
            st.warning('Face not found', icon='⚠️')

    min_similar = st.slider('Minimum Similarity', value=70, min_value=10, max_value=99, step=5)

    if embs is not None:
        for img in src_imgs:
            src_embs = get_image_embeddings(np.array(img))
            if src_embs is not None:
                cosine = (src_embs @ embs.T).flatten() * 100
                if cosine.max() >= min_similar:
                    st.image(img, f'{round(cosine.max())}%')


def read_zip_file(uploaded_file):
    cols = st.columns(4)
    imgs = []  # This line was likely missing in the image
    with ZipFile(uploaded_file, 'r') as zip:
        for filename in zip.namelist():
            if filename.endswith('png') or filename.endswith('jpg') or filename.endswith('jpeg'):
                with zip.open(filename) as f:
                    img = Image.open(f)
                    imgs.append(img)
                    st.image(img, f'{filename[-20:-4]}')

def processor_text(text):
    # This function should process the text input for the text model
    # For simplicity, we will just return the text as a list
    return [text]

def model_text():
    # This function should return the text model and its tokenizer
    # For simplicity, we will return a dummy model and tokenizer
    class DummyModel:
        def encode(self, text_data):
            # Dummy embedding for the text
            return np.random.rand(1, 512), np.random.rand(512)

    return DummyModel(), None

def text_search(src_imgs, src_embs):
    col1, col2 = st.columns(2)
    with col1:
        text = st.text_area('Image Description')
    with col2:
        min_cosine = st.slider('Level of Similarity (%)', value=30, min_value=10, max_value=100, step=5)

    if len(text) > 0:
        text_data = processor_text(text)
        _, text_embedding = model_text.encode(text_data)
        text_embedding = text_embedding.flatten() / norm(text_embedding)
        cosine = (src_embs @ text_embedding) * 100
        ids = np.where(cosine >= min_cosine)[0]
        if len(ids) == 0:
            st.info('Not found')
        else:
            result = [(cosine[i], i) for i in ids]
            result.sort(reverse=True)

            j = 0
            st.success('Result')
            cols = st.columns(3)
            for cosine, i in result:
                with cols[j % 3]:
                    st.image(src_imgs[i], f'{round(cosine)}%')
                j += 1

                    
def main():
    st.title('🖼 Image Search')
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Images", type=['zip'])

    if uploaded_file is None:
        st.text('👈 Please upload your images')        
    else:
        with st.expander('Source Images'):
            read_zip_file(uploaded_file)

        tab1, tab2 = st.tabs(['🙂 Face Search', '📄 Text Search'])
        with tab1:
            st.write('Search Face')
        with tab2:
            st.write('Search Text')



# def read_zip(uploaded_file):
#     try:
#         with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
#             image_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
#             if image_files:
#                 st.subheader("Hình ảnh trong file ZIP:")
#                 cols = st.columns(4)  # Hiển thị ảnh theo lưới 4 cột
#                 for i, img_file in enumerate(image_files):
#                     try:
#                         with zip_ref.open(img_file) as img_file_in_zip:
#                             img = Image.open(io.BytesIO(img_file_in_zip.read()))
#                             cols[i % 4].image(img, caption=img_file, use_container_width=True)
#                     except Exception as e:
#                         st.error(f"Lỗi khi đọc ảnh '{img_file}': {e}")
#             else:
#                 st.info("Không tìm thấy hình ảnh nào trong file ZIP.")

#     except zipfile.BadZipFile:
#         st.error("File đã tải lên không phải là file ZIP hợp lệ.")
#     except Exception as e:
#         st.error(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    main()