# search-photos-by-photo

Search photos by an uploaded photo using deepface

This is the project 6. Demo: https://www.youtube.com/watch?v=1v-49mTu0wE

## Authors

- [@LisaTheTechnologue](https://www.github.com/LisaTheTechnologue)

## Tech Stack

- Thư viện sử dụng: numpy, OpenCV, Pillow, Streamlit, uform, deepface.

- Chức năng cơ bản: tìm kiếm ảnh theo ảnh gương mặt & theo mô tả bằng text.

- Chức năng nâng cao: tìm kiếm nhiều người trong ảnh

- Những kỹ năng học được:

  - Kỹ thuật xử lý file zip.

  - Kỹ thuật lấy đặc trưng của ảnh & văn bản bằng thư viện uform.

  - Kỹ thuật lấy đặc trưng gương mặt thư viện deepface.

  - Tìm kiếm ảnh bằng các kỹ thuật chuẩn hoá vector & nhân ma trận.

## Features

- Compare uploaded photo to image folder

## Flowchart

![flow chart](https://github.com/LisaTheTechnologue/search-photos-by-photo/main/mc4ia-milestone.drawio.png?raw=true)

## Appendix

### Steps to setup

1. Setup

- Python

- IDE (VSCode, PyCharm, Jupyter, …)

- Các thư viện (Numpy, Pandas, Scikit-learn …)

2. Tạo Github repo

- Thêm thông tin cơ bản vào file README.h (tên Project, danh sách thành viên, các chức năng chính …)

- Tạo các file Python (.py)

- Thêm các file dataset (nếu có)

3. Xây dựng các chức năng theo các Milestone (được thông báo trong module Post-Class)

- Session 1: Lập danh sách các chức năng và vẽ flow-chart cho từng chức năng (VD: đọc dữ liệu, huấn luyện mô hình, …)

- Session 3: Đọc dữ liệu, hiển thị giao diện cơ bản

- Session 5: Tạo dữ liệu

- Session 7: Phân loại, Phân nhóm

#### Thêm chức năng tìm kiếm ảnh theo gương mặt:

- Tìm hiểu hàm `represent` của thư viện _deepface_.
- Dùng hàm `represent` để biểu diễn gương mặt cần tìm & các gương mặt có trong file zip.
- Tính cosine similarity giữa các gương mặt & hiển thị kết quả.

- Session 8-9-10: Dự báo, vẽ các đồ thị

### Presentation

1. Giới thiệu Project

- Tính năng chính
- Tính năng mở rộng

2. Mô tả Dataset

- Nguồn
- Cách lưu trữ, đọc dataset
- Số lượng sample, số class, số sample/mỗi class
- Số đặc trưng, chi tiết từng đặc trưng

3. Mô tả model

- Thư viện sử dụng
- Số lớp, chi tiết từng lớp

4. Kết quả huấn luyện

- Số epoch, test size
- Loss, Accuracy trên Train Set & Test Set

5. Hướng cải tiến/phát triển Project (nếu có)
6. Demo các chức năng

- Lưu ý chuẩn bị sẵn các model để không mất thời gian training

### Markdown cheatsheet

https://github.com/tchapi/markdown-cheatsheet/blob/master/README.md

### Sample code

#### S3

```python
def main():
    st.title('🖼 Image Search')
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Images", type=['zip'])

    if uploaded_file is None:
        st.text('👈 Please upload your images')
    else:
        with st.expander('Source Images'):
            st.write('List Image')
        tab1, tab2 = st.tabs(['🙂 Face Search', '📄 Text Search'])
        with tab1:
            st.write('Search Face')
        with tab2:
            st.write('Search Text')
```

#### S7

Hàm represent của thư viện deepface để biểu diễn gương mặt cần tìm và các gương mặt có trong file zip
Tính cosine sim giữa các gương mặt và kết quả

```python
def read_zip_file(uploaded_file):
    cols = st.columns(4)
    with ZipFile(uploaded_file, 'r') as zip:
        for filename in zip.namelist():
            if filename.endswith('png') or filename.endswith('jpg') or filename.endswith('jpeg'):
                with zip.open(filename) as f:
                    img = Image.open(f)
                    imgs.append(img)
                    st.image(img, f'{filename[-20:-4]}')
```

#### S12

Tìm kiếm ảnh theo text

```python
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
```
