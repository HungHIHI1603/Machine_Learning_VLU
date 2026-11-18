![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)

# Machine Learning — Học Máy Và Ứng Dụng

* **Họ và tên:** Nguyễn Phi Hùng
* **MSSV:** 2474802010129
* **Môn:** Học Máy Và Ứng Dụng
* **LHP:** 261_71ITAI41203_0101
* **GVHD:** ThS. Nguyễn Thái Anh ([GitHub](https://github.com/AnhNguyenVLU))

## Bài thực hành

| Bài | Chủ đề | Thư mục |
| --- | --- | --- |
| 01 | Hồi quy tuyến tính (dữ liệu giá nhà) | [01. Linear Regression](lab1/) |
| 02 | Hồi quy Logistic (dự đoán kết quả học tập) | [02. Logistic Regression](lab2/) |
| 03 | Naive Bayes (định lý Bayes và phân loại tin nhắn) | [03. Naive Bayes](lab3/) |
| 04 | K láng giềng gần nhất (KNN) | [04. KNN](lab4/) |
| 05 | Cây quyết định và Rừng ngẫu nhiên | [05. Decision Tree & Random Forest](lab5/) |
| 06 | Máy vectơ hỗ trợ (SVM) | [06. SVM](lab6/) |
| 07 | Đánh giá mô hình học máy | [07. Model Evaluation](lab7/) |
| 08 | Phân cụm K-Means | [08. K-Means](lab8/) |
| 09 | Mạng nơ-ron nhiều lớp (MLP) | [09. MLP](lab9/) |

## Cấu trúc

```text
Machine_Learning_VLU/
├── lab1/
│   ├── lab1.ipynb          # Notebook hồi quy tuyến tính
│   └── gia_nha.csv         # Dữ liệu giá nhà
├── lab2/
│   ├── lab2.ipynb          # Notebook hồi quy Logistic
│   └── sinh_vien.csv       # Dữ liệu sinh viên
├── lab3/
│   ├── lab3.ipynb          # Notebook Naive Bayes
│   ├── sinh_vien.csv       # Dữ liệu sinh viên
│   └── tin_nhan.csv        # Dữ liệu tin nhắn
├── lab4/
│   ├── Lab4.ipynb          # Notebook KNN
│   ├── gia_nha.csv
│   ├── hai_vang_trang.csv
│   └── sinh_vien_mo_rong.csv
├── lab5/
│   ├── code/               # Các bài thực hành Python
│   ├── data/               # Dữ liệu giá nhà và khách vay
│   ├── images/             # Hình ảnh kết quả bài tập
│   └── Lab05_CayQuyetDinh_RungNgauNhien.pdf
├── lab6/
│   ├── Lab6.ipynb          # Notebook SVM
│   ├── Lab06_SVM.pdf       # Tài liệu thực hành
│   ├── sinh_vien.csv
│   ├── sinh_vien_mo_rong.csv
│   ├── hai_vang_trang.csv
│   ├── output.png
│   └── anh_sai_*.png       # Hình ảnh dự đoán sai
├── lab7/
│   ├── code/               # Các bước thực hành đánh giá mô hình
│   ├── data/               # Dữ liệu thực hành
│   ├── baitap/             # Bài tập bổ sung
│   ├── lab07/              # Tài liệu và nội dung bài tập
│   ├── Lab7_Danh_Gia_Mo_Hinh.ipynb
│   ├── Lab07_DanhGiaMoHinh.pdf
│   ├── gia_nha.csv
│   └── roc.png
├── lab8/
│   ├── code/               # Các bước thực hành K-Means
│   ├── baitap/             # Các bài tập bổ sung
│   ├── data/               # Các bộ dữ liệu thực hành
│   ├── Lab8_KMeans.ipynb
│   ├── Lab08_KMeans.pdf
│   ├── anh_mau_goc.png
│   ├── anh_mau_16_mau.png
│   └── output.png
├── lab9/
│   ├── Lab9_MLP.ipynb      # Notebook mạng nơ-ron nhiều lớp
│   ├── README.htm          # Tài liệu bài Lab
│   ├── gia_nha.csv
│   ├── hai_vang_trang.csv
│   ├── sinh_vien_mo_rong.csv
│   ├── xor.csv
│   └── toc_do_hoc.png
└── README.md
```

## Cách chạy

Clone repository:

```sh
git clone https://github.com/HungHIHI1603/Machine_Learning_VLU.git
cd Machine_Learning_VLU
```

Cài đặt các thư viện cần thiết:

```sh
python -m pip install numpy pandas matplotlib scikit-learn jupyter
```

Mở notebook bằng Jupyter:

```sh
jupyter notebook
```

Chọn thư mục Lab cần thực hành và mở tệp `.ipynb` tương ứng. Một số notebook hiện sử dụng đường dẫn dữ liệu của Google Colab (ví dụ `/content/` hoặc Google Drive); cần điều chỉnh đường dẫn cho phù hợp nếu chạy trên máy cá nhân.

## Nội dung kiến thức

- **Linear Regression:** Tìm hiểu mối quan hệ giữa diện tích và giá nhà, xây dựng đường hồi quy và dự đoán giá trị.
- **Logistic Regression:** Sử dụng hàm sigmoid và hồi quy Logistic để dự đoán khả năng sinh viên qua môn.
- **Naive Bayes:** Tìm hiểu định lý Bayes, xác suất hậu nghiệm và biểu diễn dữ liệu văn bản để phân loại tin nhắn.
- **KNN:** Tính khoảng cách, lựa chọn số láng giềng K và phân loại dữ liệu bằng phương pháp láng giềng gần nhất.
- **Decision Tree & Random Forest:** Tìm hiểu chỉ số Gini, cách phân chia nhánh, xây dựng cây quyết định và rừng ngẫu nhiên.
- **SVM:** Tìm hiểu siêu phẳng phân chia, lề, vectơ hỗ trợ, tham số C và chuẩn hóa dữ liệu.
- **Model Evaluation:** Thực hành ma trận nhầm lẫn, Accuracy, Precision, Recall, F1-score, ngưỡng dự đoán và ROC-AUC.
- **K-Means:** Tự cài đặt thuật toán, phân khúc khách hàng, chọn số cụm K và ứng dụng vào nén ảnh.
- **MLP:** Tìm hiểu lan truyền tiến, lớp ẩn, hàm kích hoạt, bài toán XOR, kích thước mạng và điều chuẩn.

## Kết luận

Qua 9 bài thực hành, sinh viên từng bước tiếp cận quy trình xây dựng một mô hình học máy: tìm hiểu dữ liệu, lựa chọn thuật toán, huấn luyện, dự đoán và đánh giá kết quả.

Các bài Lab giúp củng cố kiến thức về hồi quy, phân loại, phân cụm và mạng nơ-ron; đồng thời rèn luyện kỹ năng sử dụng Python, NumPy, Pandas, Matplotlib và Scikit-learn để giải quyết các bài toán với dữ liệu thực tế.

---

**Repository:** [Machine_Learning_VLU](https://github.com/HungHIHI1603/Machine_Learning_VLU)  
**Course:** Machine Learning — Học Máy Và Ứng Dụng  
**University:** Văn Lang University (VLU)  
**Labs:** 01 – 09
