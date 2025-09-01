# search-photos-by-photo

Demo: https://www.youtube.com/watch?v=1v-49mTu0wE

---

## Authors

- [@LisaTheTechnologue](https://www.github.com/LisaTheTechnologue)

<!-- TOC -->
<!-- TOC END -->

---

## Tech Stack

- Thư viện sử dụng: numpy, OpenCV, Pillow, Streamlit, uform, deepface.

- Chức năng cơ bản: tìm kiếm ảnh theo ảnh gương mặt & theo mô tả bằng text.

- Chức năng nâng cao: tìm kiếm nhiều người trong ảnh

- Những kỹ năng học được:

  - Kỹ thuật xử lý file zip.

  - Kỹ thuật lấy đặc trưng của ảnh & văn bản bằng thư viện uform.

  - Kỹ thuật lấy đặc trưng gương mặt thư viện deepface.

  - Tìm kiếm ảnh bằng các kỹ thuật chuẩn hoá vector & nhân ma trận.

---

## Features

- Upload 'data.zip' folder: contains all images to search

- First Tab: Image search

- Take photo then search in uploaded data folder by Level of Similarity

- Upload image then search in uploaded data folder by Level of Similarity

- Show 'Result' and list of images OR 'Not Found' if no result

- Compare uploaded photo to image folder

- Second Tab: Text Search

---

## Flowchart

![flow chart](https://github.com/LisaTheTechnologue/search-photos-by-photo/blob/main/mc4ia-milestone-Page-2.png)

---

## Appendix

---

### DeepFace

#### Facenet

“128 chiều” trong FaceNet embedding

- Khi DeepFace (với backend FaceNet) trích xuất đặc trưng của một khuôn mặt, nó tạo ra một vector có 128 số thực.

- Nghĩa là mỗi khuôn mặt được biểu diễn trong không gian toán học 128 chiều (gọi là embedding space).

- Kết hợp cả 128 số này → mô tả duy nhất cho khuôn mặt.

- Hai khuôn mặt giống nhau → vector embedding gần nhau trong không gian 128D.

- Hai khuôn mặt khác nhau → vector embedding cách xa nhau.

#### Method chính

**🔹 1. `verify(img1, img2)`**

- **Chức năng:**
  So sánh **hai ảnh khuôn mặt** → trả về:

  - `verified` (True/False)
  - `distance` (khoảng cách giữa hai vector embedding)
  - `model` dùng để so sánh

- **Ưu điểm:**

  - Dùng trực tiếp khi chỉ muốn so sánh **1 ảnh A với 1 ảnh B**.
  - Có sẵn logic True/False (cùng người hay khác người).

- **Nhược điểm:**

  - Không phù hợp khi cần so sánh **1 ảnh với nhiều ảnh** (sẽ phải gọi vòng lặp, hơi chậm).
  - Không lưu lại embedding để dùng lại sau → tốn thời gian nếu ảnh nhiều.

- **Ứng dụng:**
  Nếu bạn chỉ cần **check một cặp ảnh** thì dùng `verify` là đơn giản nhất.

** 🔹 2. `represent(img)`**

- **Chức năng:**
  Trích xuất **embedding vector** (ví dụ 128D, 512D tùy model).

- **Ưu điểm:**

  - Lấy embedding 1 lần, sau đó so sánh bằng khoảng cách (cosine, euclidean, etc.).
  - Rất phù hợp cho **search/retrieval** (tìm trong một thư viện ảnh).
  - Nhanh hơn `verify` nhiều khi số ảnh lớn (vì không phải chạy model nhiều lần).

- **Nhược điểm:**

  - Cần tự viết hàm so sánh (cosine similarity, L2 distance…).
  - Không trả về “verified=True/False” sẵn.

- **Ứng dụng:**
  Nếu bạn muốn xây **face search engine** (giống Google Images hoặc FaceNet demo), thì `represent` là lựa chọn chuẩn nhất.

** 🔹 3. `analyze(img)`**

- **Chức năng:**
  Dự đoán thuộc tính khuôn mặt: tuổi, giới tính, cảm xúc, sắc tộc.
- **Không phù hợp** cho bài toán tìm ảnh giống nhau.

** 🔹 4. `find(img, db_path)`**

- **Chức năng:**
  Cho một ảnh query, tìm ảnh tương đồng nhất trong database (thư mục ảnh).
- **Ưu điểm:**

  - Tự động load ảnh trong thư mục → không cần code vòng lặp.
  - Có sẵn cơ chế tìm nearest neighbor.

- **Nhược điểm:**

  - Bị ràng buộc vào DB path → khó tùy biến nếu muốn xử lý ảnh upload tạm thời.

- **Ứng dụng:**
  Dùng nếu bạn muốn một **công cụ nhanh gọn** để tìm kiếm ảnh trong thư mục.

** Có 2 cách cho bài toán này:**

1. **Nhanh gọn (ít code):** dùng `DeepFace.find(query_img, db_path)`
   → Tự động tìm ảnh giống trong folder.
   → Nhưng bất tiện nếu ảnh chỉ nằm trong **Streamlit uploader** chứ không phải folder thật.

2. **Tối ưu & linh hoạt:** dùng `represent`

   - Bước 1: Trích xuất embedding cho tất cả ảnh trong folder **1 lần**.
   - Bước 2: Trích xuất embedding cho ảnh query.
   - Bước 3: Tính cosine similarity giữa query và từng ảnh.
   - Bước 4: Lọc theo slider % Similarity.

---

### Presentation

#### QA:

1. Đây là gì `np.array(embs[0]['embedding'])`

- `embs[0]`: Lấy phần tử đầu tiên trong danh sách (trường hợp ảnh chỉ có 1 khuôn mặt).
- Truy cập vào key "embedding" để lấy ra vector đặc trưng (embedding vector) của gương mặt.
- Các key quan trọng bên cạnh embedding
  - `facial_area`: Là một dict chứa toạ độ bounding box khuôn mặt: x, y, w, h. 👉 Dùng để vẽ khung mặt trên ảnh, cắt ảnh khuôn mặt ra, hoặc highlight phần detect.
  - `face_confidence`: Giá trị độ tin cậy của detector khi phát hiện khuôn mặt (thường từ 0 → 1). 👉 Dùng để lọc bỏ khuôn mặt kém rõ ràng (ví dụ < 0.9 thì bỏ).

2. Nếu như muốn tìm 1 người trong set ảnh có m người thì dùng bài toán nào:

- Bài toán (m,128) \* (1,128) --> Broadcast, chuẩn hóa rồi tính cosine similarity

3. Nếu như muốn tìm n người trong set ảnh có m người thì dùng bài toán nào:

- Bài toán (m,128) \* (n,128) --> norm --> chuyển (n,128) thành ma trận transpose (128,n) --> nhân ma trận

---

## Markdown cheatsheet

https://github.com/tchapi/markdown-cheatsheet/blob/master/README.md

---

## Sample code

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
