# Face & Text Search App 🎯

Ứng dụng cuối khóa **Math for AI**: tìm kiếm ảnh bằng **gương mặt** hoặc bằng **mô tả văn bản**.  
Xây dựng bằng **Python + Streamlit**, sử dụng hai mô hình AI mạnh mẽ:

- **DeepFace (Facenet)** → trích xuất embedding gương mặt để tìm kiếm ảnh giống nhau.
- **UForm** → encode cả văn bản tiếng Anh và ảnh vào cùng không gian vector để tìm kiếm ảnh theo text.

---

## 🚀 Tính năng chính

1. **Face Search**

   - Người dùng chụp ảnh hoặc upload ảnh gương mặt.
   - Ứng dụng so sánh embedding với dataset ảnh và trả về các gương mặt tương tự.

2. **Text Search**
   - Người dùng nhập mô tả ngắn gọn bằng tiếng Anh.
   - Ứng dụng so sánh embedding văn bản với embedding ảnh trong dataset và trả về ảnh phù hợp.

---

## 🗂️ Cấu trúc thư mục

.
├── app.py # Main Streamlit app (chứa tabs)
├── face_search.py # Module cho Face Search
├── text_search.py # Module cho Text Search
├── image_utils.py # Xử lý ảnh, embeddings, cosine similarity
├── requirements.txt # Danh sách thư viện cần cài
└── README.md # Tài liệu hướng dẫn

---

## ⚙️ Cài đặt

1. Tạo virtual environment và activate (Windows):

```bash
python -m venv venv
venv\Scripts\activate
```

2. Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

3. Chạy ứng dụng

```bash
streamlit run app.py
```

Mở trình duyệt tại http://localhost:8501

---

## Cách hoạt động (ngắn gọn)

- DeepFace (Facenet)

  - Input: ảnh gương mặt.

  - Output: vector embedding 128 chiều.

  - So sánh bằng cosine similarity để tìm ảnh giống.

- UForm

  - Input: ảnh hoặc text tiếng Anh.

  - Output: vector embedding 512 chiều trong cùng không gian ngữ nghĩa.

  - So sánh text vs image bằng cosine similarity.

- Cosine Similarity

  - Đo độ giống giữa hai vector bằng cosine của góc giữa chúng.

  - Giá trị gần 1 → rất giống, gần 0 → không liên quan.

---

## Nhóm thực hiện

- Dự án cuối khóa Math for AI.
  - Sinh viên: Tran Chung Bao Ngan
  - Giảng viên hướng dẫn: Nguyen Viet Ha (CotAi)
