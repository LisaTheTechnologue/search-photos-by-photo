from image_utils import get_image_embeddings, read_zip_file
from face_search import face_search
from text_search import text_search
import streamlit as st
import numpy as np

def main():
    st.title('🖼 Image Search')
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Images", type=['zip'])

    if uploaded_file is None:
        st.text('👈 Please upload your images')        
    else:
        with st.expander('Source Images'):
            src_imgs = read_zip_file(uploaded_file)

        tab1, tab2 = st.tabs(['🙂 Face Search', '📄 Text Search'])
        with tab1:
            face_search(src_imgs)
        
        with tab2:
            text_search(src_imgs)

if __name__ == "__main__":
    main()