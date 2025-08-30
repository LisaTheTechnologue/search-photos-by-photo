from image_utils import read_zip_file
from face_search import face_search
from text_search import text_search
import streamlit as st

def main():
    st.title('🖼 Face & Text Search')

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Images", type=['zip'])

    if uploaded_file is None:
        st.text('👈 Please upload your images')        
    else:
        with st.expander('Source Images'):
            src_imgs = read_zip_file(uploaded_file)

        # Tạo 2 tab song song
        tab1, tab2 = st.tabs(["🔍 Face Search", "🔤 Text Search"])
        with tab1:
            face_search(src_imgs)
        with tab2:
            text_search(src_imgs)

if __name__ == "__main__":
    main()
