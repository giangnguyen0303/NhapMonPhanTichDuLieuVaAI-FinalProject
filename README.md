# NhapMonPhanTichDuLieuVaAI — Final Project (IMDb Analytics & AI)

Ứng dụng web dự đoán và phân cụm phim IMDb dựa trên dữ liệu Kaggle *IMDb Dataset 2024*.
Sinh viên thực hiện: Nhập môn Phân tích Dữ liệu & AI — **Thành viên 5 (Tân)** phụ trách
**giao diện, tích hợp và tổng hợp**.

## 1. Cấu trúc thư mục

```
Final_Project/
├── app.py                       ← Giao diện chính Streamlit (thành viên 5)
├── app/
│   ├── __init__.py
│   ├── components.py            ← Biểu đồ, bảng chỉ số, khối hiển thị kết quả
│   └── utils.py                 ← Kiểm tra dữ liệu nhập, định dạng kết quả, dữ liệu demo
├── src/
│   ├── model_prediction.py      ← Nạp mô hình + dựng đặc trưng + dự đoán
│   ├── data_loader.py / data_cleaning.py / feature_engineering.py / ...
├── models/
│   ├── classification/classification_pipeline.pkl   (pipeline Decision Tree, notebook 07)
│   ├── regression/regression_pipeline.pkl           (pipeline Linear Regression, notebook 06)
│   └── clustering/pca_model.pkl + kmeans_model.pkl  (PCA 9 chiều + KMeans k=4, notebook 05)
├── data/
│   ├── raw/                     ← 3 file CSV gốc từ Kaggle
│   └── processed/               ← Dữ liệu đã làm sạch & feature engineering
├── notebooks/                   ← 01–07 (load, clean, EDA, features, model)
├── report/screenshots/          ← Ảnh chụp demo ứng dụng
└── requirements.txt
```

## 2. Cài đặt & chạy

```bash
# (khuyến nghị) tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate        # Windows

pip install -r requirements.txt
streamlit run app.py 
hoặc
Chạy file chay_app.bat
```
App mở tại `http://localhost:8501`. Trong sidebar chọn chức năng:
**Trang chủ · Phân loại · Hồi quy · Phân cụm**.

### Chế độ demo (tiền điền dữ liệu mẫu + tự động dự đoán)

| Trang | URL |
| --- | --- |
| Phân loại | `http://localhost:8501/?page=classification&demo=1` |
| Hồi quy | `http://localhost:8501/?page=regression&demo=1` |
| Phân cụm | `http://localhost:8501/?page=clustering&demo=1` |

## 3. Chức năng của ứng dụng

| Trang | Nhập | Kết quả |
| --- | --- | --- |
| 🌲 **Phân loại** | Tên phim, năm, thời lượng, MetaScore, thể loại, chứng nhận, dàn diễn viên | Nhãn **High (≥ 7.0)** / **Low (< 7.0)**, xác suất từng lớp, bảng 29 đặc trưng đầu vào |
| 📈 **Hồi quy** | Năm, thời lượng, MetaScore, chứng nhận, thể loại chính | Điểm IMDb dự đoán (gauge + histogram phân bố thực tế, so với trung bình dữ liệu) |
| 🔍 **Phân cụm** | Tên phim, năm, thời lượng, MetaScore, điểm IMDb, thể loại, chứng nhận | Cụm phim (k=4) kèm tên diễn giải, hồ sơ các cụm, vị trí phim trên không gian PCA |

Trang chủ hiển thị **trạng thái nạp mô hình**, chỉ số tổng hợp của ba mô hình,
phân bố dữ liệu và luồng xử lý.

## 4. Mô hình tích hợp

| Mô hình | File | Kiến trúc | Chỉ số (tập test / đánh giá) |
| --- | --- | --- | --- |
| Classification | `models/classification/classification_pipeline.pkl` | Decision Tree (depth=5, leaf=50, class_weight=balanced) + imputer/scaler/one-hot | Accuracy 72.85%, F1-macro 0.7197, ROC-AUC 0.7802 |
| Regression | `models/regression/regression_pipeline.pkl` | Linear Regression + imputer/scaler/one-hot | RMSE 0.6935, R² 0.3592 |
| Clustering | `models/clustering/pca_model.pkl`, `kmeans_model.pkl` | StandardScaler → PCA (9 chiều, 90.7% phương sai) → KMeans (k=4) | Silhouette 0.192, Davies-Bouldin 1.548 |

Tên 4 cụm (diễn giải theo hồ sơ trung bình): *Giải trí đại chúng điểm thấp ·
Đại tác phẩm dài, điểm cao · Phim cổ điển (trước 1996) · Phim hiện đại chất lượng khá*.

### Ghi chú kỹ thuật

- **Nạp mô hình**: `src/model_prediction.py` nạp file `.pkl` bằng `joblib` (cache qua
  `lru_cache`). Pipeline phân loại và hồi quy gồm cả tiền xử lý (imputer, scaler,
  one-hot) nên đầu vào chỉ cần một `DataFrame` đúng tên cột.
- **Thiếu file mô hình**: nếu `regression_pipeline.pkl` / `pca_model.pkl` +
  `kmeans_model.pkl` chưa có, module tự huấn luyện lại đúng công thức của notebook 06/05
  và ghi file vào `models/` (số liệu khớp notebook: RMSE 0.6935, silhouette 0.192, k=4).
- **Đầu vào phân cụm** dùng 20 đặc trưng như notebook 05 (4 biến số chuẩn hoá +
  one-hot thể loại gom ≥1.5% tần suất và nhóm chứng nhận R/PG-13/PG/G/Khác); thể loại
  chính của phim gom vào `Other` nếu không thuộc nhóm phổ biến.
- **Không rò rỉ nhãn**: mô hình phân loại không nhận điểm IMDb làm đầu vào;
  nhãn `rating_class` được xác định trước theo ngưỡng 7.0.

## 5. Ảnh chụp demo

Ảnh chụp ứng dụng thực tế nằm trong `report/screenshots/`:

| File | Nội dung |
| --- | --- |
| `streamlit_home.png` | Trang chủ tổng quan |
| `classification_result.png` | Kết quả trang phân loại |
| `regression_result.png` | Kết quả trang hồi quy |
| `clustering_result.png` | Kết quả trang phân cụm |

## 6. Nguồn dữ liệu

- Kaggle: [IMDb Dataset 2024 (Updated)](https://www.kaggle.com/datasets/parthdande/imdb-dataset-2024-updated)
