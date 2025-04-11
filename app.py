import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import zipfile
import io

def main():
    st.title('🖼 Image Search')
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Images", type=['zip'])

    if uploaded_file is None:
        st.text('👈 Please upload your images')        
    else:
        with st.expander('Source Images'):
            read_zip(uploaded_file)

        tab1, tab2 = st.tabs(['🙂 Face Search', '📄 Text Search'])
        with tab1:
            st.write('Search Face')
        with tab2:
            st.write('Search Text')

def read_zip(uploaded_file):
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            image_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            if image_files:
                st.subheader("Hình ảnh trong file ZIP:")
                cols = st.columns(4)  # Hiển thị ảnh theo lưới 4 cột
                for i, img_file in enumerate(image_files):
                    try:
                        with zip_ref.open(img_file) as img_file_in_zip:
                            img = Image.open(io.BytesIO(img_file_in_zip.read()))
                            cols[i % 4].image(img, caption=img_file, use_container_width=True)
                    except Exception as e:
                        st.error(f"Lỗi khi đọc ảnh '{img_file}': {e}")
            else:
                st.info("Không tìm thấy hình ảnh nào trong file ZIP.")

    except zipfile.BadZipFile:
        st.error("File đã tải lên không phải là file ZIP hợp lệ.")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    main()