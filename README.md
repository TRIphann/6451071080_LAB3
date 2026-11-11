# LAB 03 — Khai Phá Dữ Liệu (Data Mining)

**Họ tên:** Phan Công Trí  
**MSSV:** 6451071080  
**Môn học:** Khai phá dữ liệu — Bài Lab 03  
**Kho lưu trữ:** [TRIphann/6451071080_LAB3](https://github.com/TRIphann/6451071080_LAB3)

---

## 📋 Nội dung bài Lab

Bài Lab 03 tập trung vào các kỹ thuật **học có giám sát** (supervised learning),
**học không giám sát** (unsupervised learning) và **đánh giá mô hình** trên các bộ dữ liệu
kinh điển: Iris, MNIST, Mall Customers, Breast Cancer Wisconsin.

| # | Bài | Nội dung chính | Dữ liệu sử dụng |
|---|-----|-----------------|----------------|
| 1 | [Bài 1](#-bài-1--phân-lớp-với-nhiều-mô-hình) | Phân lớp với nhiều mô hình: Decision Tree, Naive Bayes, KNN, Neural Network | Iris (sklearn) |
| 2 | [Bài 2](#-bài-2--mạng-nơ-ron-và-mnist) | Xây dựng & huấn luyện mạng nơ-ron phân lớp ảnh viết tay | MNIST (Keras) |
| 3 | [Bài 3](#-bài-3--k-fold-cross-validation-với-mnist) | K-fold cross validation (k=10) trên mạng nơ-ron MNIST | MNIST (Keras) |
| 4 | [Bài 4](#-bài-4--k-fold-cross-validation-với-iris) | K-fold cross validation (k=10) với đầy đủ chỉ số Accuracy/Precision/Recall/F1 | Iris (sklearn) |
| 5 | [Bài 5](#-bài-5--gom-cụm-k-means-và-gom-cụm-các-loại-khác) | Gom cụm K-Means: MNIST, Breast Cancer, phân tích PCA & Silhouette | MNIST, Breast Cancer |
| 6 | [Bài 6](#-bài-6--nén-ảnh-bằng-k-means) | Nén ảnh (color quantization) bằng K-Means | Ảnh mẫu |

---

## 📁 Cấu trúc thư mục

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── 6451071080_LAB3_Bai1.ipynb
├── 6451071080_LAB3_Bai2.ipynb
├── 6451071080_LAB3_Bai2_Trang18_19.ipynb
├── 6451071080_LAB3_Bai3.ipynb
├── 6451071080_LAB3_Bai4.ipynb
├── 6451071080_LAB3_Bai5_Trang26.ipynb
├── 6451071080_LAB3_Bai5_Trang29.ipynb
└── 6451071080_LAB3_Bai6_Trang26.ipynb
```

> **Lưu ý:** File đề bài gốc ở dạng PDF (`6451071080_LAB3_N1.pdf`) được `.gitignore` loại trừ
> và **không** đẩy lên GitHub theo yêu cầu. Bạn vẫn có thể tải đề từ lớp học.

---

## 📓 Mô tả chi tiết từng bài

### 📘 Bài 1 — Phân lớp với nhiều mô hình

Notebook: `6451071080_LAB3_Bai1.ipynb`

| Câu | Nội dung |
|-----|----------|
| a | Đọc bộ dữ liệu **Iris** từ `sklearn.datasets.load_iris()` |
| b | In tên các thuộc tính, thuộc tính phân lớp, số mẫu và số thuộc tính |
| c | **Chuẩn hóa** dữ liệu về đoạn `[0, 1]` ( chia theo giá trị lớn nhất từng cột) |
| d | Chia tập **train/test** theo tỷ lệ **8:2** (`random_state=42`) |
| e | Huấn luyện **Decision Tree Classifier** + đánh giá (Accuracy, Report, Confusion Matrix) |
| f | Huấn luyện **Naive Bayes (GaussianNB)** |
| g | Huấn luyện **K-Nearest Neighbors Classifier** |
| h | Xây dựng **Neural Network**: 2 hidden layer (10, 20) + 1 output layer (3) |
| i | **Dự đoán** phân lớp cho 03 mẫu dữ liệu mới (đã chuẩn hóa) |
| j | **Nhận xét & đánh giá** hiệu suất giữa 4 mô hình |

### 📘 Bài 2 — Mạng nơ-ron và MNIST

Notebook: `6451071080_LAB3_Bai2.ipynb`

| Câu | Nội dung |
|-----|----------|
| a–d | Đọc tập train/test **MNIST** từ Keras, in thông tin dữ liệu và **chuẩn hóa** pixel về `[0,1]` |
| e | Xây dựng mạng nơ-ron: `Flatten → Dense(128, relu) → Dropout(0.2) → Dense(10, softmax)` |
| f | **Biên dịch** mô hình (Adam + sparse categorical crossentropy) |
| g | **Huấn luyện** mô hình (5 epochs) |
| h | Vẽ biểu đồ **accuracy / loss** sau mỗi epoch (train & validation) |
| i | Dự đoán phân lớp cho tập test |
| j | Tính **Accuracy** trên tập test |
| k | Tính **Precision, Recall, F1-score** (macro/micro/weighted) |
| l | In **Confusion Matrix** |
| m | In kết quả phân lớp của **05 ảnh đầu tiên** trong tập test |
| n | Kết luận & nhận xét |

### 📘 Bài 2 (trang 18–19) — Gom cụm khách hàng

Notebook: `6451071080_LAB3_Bai2_Trang18_19.ipynb`

| Câu | Nội dung |
|-----|----------|
| 1 | Nhập dữ liệu **Mall Customers**, thống kê mô tả, xử lý dữ liệu thiếu |
| 2 | **Histogram** phân phối: Tuổi, Thu nhập, Số điểm mua sắm |
| 3 | Thống kê số mẫu theo **Giới tính** (countplot) |
| 4 | **Scatter plot** khảo sát Tuổi & Thu nhập theo Giới tính |
| 5 | **Scatter plot** khảo sát Thu nhập & Số điểm mua sắm theo Giới tính |
| 6–8 | Gom cụm theo **Tuổi & Số điểm mua sắm** bằng **ELBOW + K-Means**, trực quan hóa & đánh giá |
| 9 | Gom cụm theo **Thu nhập & Số điểm mua sắm** — 2 trường hợp, so sánh |

### 📘 Bài 3 — K-fold cross validation với MNIST

Notebook: `6451071080_LAB3_Bai3.ipynb`

| Câu | Nội dung |
|-----|----------|
| a–c | Đọc & chuẩn hóa **MNIST** |
| d | Định nghĩa hàm **`create_model()`** |
| e | Huấn luyện bằng **k-fold cross validation (k=10)** với `shuffle=True`, `random_state=42` |
| f | Tính **Mean/Std Accuracy**, **Mean/Std Loss** qua các fold |
| g | Vẽ biểu đồ **accuracy & loss** theo từng fold |
| h | Chọn **mô hình tốt nhất** (fold có accuracy cao nhất) |
| i–l | Đánh giá trên tập test: Accuracy, Precision, Recall, F1, Confusion Matrix |

### 📘 Bài 4 — K-fold cross validation với Iris

Notebook: `6451071080_LAB3_Bai4.ipynb`

| Câu | Nội dung |
|-----|----------|
| a–c | Đọc **Iris**, chuẩn hóa, tách tập train/test |
| d | **k-fold cross validation (k=10)** với mạng nơ-ron, thu thập Precision/Recall/F1 theo từng lớp |
| e | In **Average Accuracy / Precision / Recall / F1-score** trên 10 folds |
| f–g | Trực quan hóa kết quả & **nhận xét** |

### 📘 Bài 5 — Gom cụm K-Means (trang 26 & trang 29)

Notebooks: `6451071080_LAB3_Bai5_Trang26.ipynb`, `6451071080_LAB3_Bai5_Trang29.ipynb`

**Trang 26 — MNIST:**
- Gom cụm **K-Means** trên tập ảnh viết tay, đánh giá bằng **Accuracy, Classification Report, Confusion Matrix**
- **Silhouette Score** để chọn số cụm `k` tối ưu
- Trực quan hóa & **nhận xét**

**Trang 29 — Breast Cancer Wisconsin:**
- Chuẩn hóa dữ liệu bằng **StandardScaler**
- **K-Means** + **PCA** giảm chiều & trực quan hóa
- Đánh giá bằng **Silhouette Score**, **Calinski-Harabasz**, **Davies-Bouldin**
- So sánh với kết quả phân lớp để **nhận xét**

### 📘 Bài 6 — Nén ảnh bằng K-Means

Notebook: `6451071080_LAB3_Bai6_Trang26.ipynb`

- Đọc ảnh, lấy kích thước & số kênh màu
- Chuẩn hóa pixel `[0,255] → [0,1]`, reshape 3D → 2D
- **K-Means (k=32)** để tìm 32 màu đại diện → **color quantization**
- Khôi phục & so sánh ảnh gốc / ảnh đã nén, **so sánh dung lượng file**

---

## 🛠️ Cài đặt & chạy code

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Chạy trên Google Colab

Tải các file `.ipynb` lên [Google Colab](https://colab.research.google.com/) và
**Runtime → Run all**.

### 3. Chạy cục bộ

```bash
jupyter notebook
```

Hoặc dùng VS Code (Jupyter extension), VS Code + Python Interactive window.

---

## 🔗 Về dữ liệu

| Bộ dữ liệu | Nguồn |
|-------------|-------|
| Iris | `sklearn.datasets.load_iris()` (tải sẵn) |
| MNIST | `tensorflow.keras.datasets.mnist` (tải sẵn) |
| Breast Cancer Wisconsin | `sklearn.datasets.load_breast_cancer()` (tải sẵn) |
| Mall Customers | Tải tự động từ GitHub (`SteffiPeTaffy/machineLearningAZ`) nếu chưa có file `Mall_Customers.csv` |
| Ảnh mẫu (Bài 6) | `sklearn.datasets.load_sample_image("china.jpg")` — tự sinh `dog.png` nếu thiếu |

> Các notebook đã có sẵn logic tự tải dữ liệu, nên có thể chạy trực tiếp mà không cần upload thủ công.

---

## 🧰 Thư viện sử dụng

| Nhóm | Thư viện |
|------|----------|
| Xử lý dữ liệu | `numpy`, `pandas` |
| Trực quan hóa | `matplotlib`, `seaborn`, `plotly` |
| Học máy | `scikit-learn` |
| Học sâu | `tensorflow` (Keras), `keras` |
| Thống kê & khác | `scipy`, `Pillow`, `urllib3` |

---

## 📝 Ghi chú

- Tất cả notebook đều in dòng `Họ tên: Phan Công Trí - MSSV: 6451071080` ở đầu mỗi cell để nhận diện.
- Sử dụng `random_state=42` xuyên suốt để kết quả **tái lập được**.
- Kết quả (ảnh biểu đồ) đã được lưu trong output của notebook sau khi chạy.

---

## 📄 Giấy phép

Dữ liệu thuộc về các thư viện mã nguồn mở (`scikit-learn`, `Keras/MNIST`).
