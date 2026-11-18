![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)

# Machine Learning — Học Máy Và Ứng Dụng

* **Họ và tên:** Nguyễn Phi Hùng
* **MSSV:** 2474802010129
* **Môn:** Học Máy Và Ứng Dụng
* **LHP:** 261_71ITAI41203_0102
* **GVHD:** ThS. Nguyễn Thái Anh ([GitHub](https://github.com/AnhNguyenVLU))

## Bài thực hành

| Bài | Chủ đề | Thư mục |
| --- | --- | --- |
| 01 | Hồi quy tuyến tính (dữ liệu giá nhà) | [01. Linear Regression](lab1/) |
| 02 | Hồi quy Logistic | [02. Logistic Regression](lab2/) |
| 03 | Naive Bayes | [03. Naive Bayes](lab3/) |
| 04 | K láng giềng gần nhất (KNN) | [04. KNN](lab4/) |
| 05 | Cây quyết định và Rừng ngẫu nhiên | [05. Decision Tree & Random Forest](lab5/) |
| 06 | Máy vectơ hỗ trợ (SVM) | [06. SVM](lab6/) |
| 07 | Đánh giá mô hình | [07. Model Evaluation](lab7/) |
| 08 | Phân cụm K-Means | [08. K-Means](lab8/) |
| 09 | Mạng nơ-ron nhiều lớp (MLP) | [09. MLP](lab9/) |

## Cấu trúc

```text
Machine_Learning_VLU/
├── lab1/
│   ├── lab1.ipynb
│   └── gia_nha.csv
├── lab2/
│   ├── lab2.ipynb
│   └── sinh_vien.csv
├── lab3/
│   ├── lab3.ipynb
│   ├── sinh_vien.csv
│   └── tin_nhan.csv
├── lab4/
│   ├── Lab4.ipynb
│   ├── gia_nha.csv
│   ├── hai_vang_trang.csv
│   └── sinh_vien_mo_rong.csv
├── lab5/
│   ├── code/
│   ├── data/
│   ├── images/
│   └── Lab05_CayQuyetDinh_RungNgauNhien.pdf
├── lab6/
│   ├── Lab6.ipynb
│   ├── Lab06_SVM.pdf
│   ├── hai_vang_trang.csv
│   ├── sinh_vien.csv
│   ├── sinh_vien_mo_rong.csv
│   ├── output.png
│   └── anh_sai_*.png
├── lab7/
│   ├── code/
│   ├── data/
│   ├── baitap/
│   ├── lab07/
│   ├── Lab7_Danh_Gia_Mo_Hinh.ipynb
│   ├── Lab07_DanhGiaMoHinh.pdf
│   ├── gia_nha.csv
│   ├── sinh_vien_bo_hoc.csv
│   └── roc.png
├── lab8/
│   ├── code/
│   ├── data/
│   ├── baitap/
│   ├── Lab8_KMeans.ipynb
│   ├── Lab08_KMeans.pdf
│   ├── khach_hang.csv
│   ├── hai_vang_trang.csv
│   ├── anh_mau_goc.png
│   ├── anh_mau_16_mau.png
│   └── output.png
├── lab9/
│   ├── Lab9_MLP.ipynb
│   ├── README.htm
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

Mở Jupyter Notebook:

```sh
jupyter notebook
```

Chọn thư mục Lab cần thực hành và mở tệp `.ipynb` tương ứng. Với các bài có mã nguồn Python riêng, mở tệp trong thư mục `code/` và chạy theo hướng dẫn của bài.

## Nội dung kiến thức

- **Linear Regression:** Tìm hiểu mối quan hệ giữa diện tích và giá nhà, xây dựng đường hồi quy và dự đoán giá trị.
- **Logistic Regression:** Sử dụng hồi quy Logistic để giải quyết bài toán phân loại.
- **Naive Bayes:** Tìm hiểu định lý Bayes, xác suất và phân loại dữ liệu.
- **KNN:** Phân loại dữ liệu dựa trên khoảng cách và số lượng láng giềng gần nhất.
- **Decision Tree & Random Forest:** Tìm hiểu chỉ số Gini, cách phân chia nhánh, cây quyết định và rừng ngẫu nhiên.
- **SVM:** Tìm hiểu siêu phẳng phân chia, lề và vectơ hỗ trợ.
- **Model Evaluation:** Thực hành Confusion Matrix, Accuracy, Precision, Recall, F1-score và ROC-AUC.
- **K-Means:** Tự cài đặt thuật toán, phân cụm khách hàng, lựa chọn số cụm và ứng dụng nén ảnh.
- **MLP:** Tìm hiểu lan truyền tiến, lớp ẩn, hàm kích hoạt, bài toán XOR và điều chuẩn.

## Kết luận

Qua 9 bài thực hành, sinh viên từng bước tiếp cận quy trình xây dựng mô hình học máy: tìm hiểu dữ liệu, lựa chọn thuật toán, huấn luyện, dự đoán và đánh giá kết quả.

Các bài Lab giúp củng cố kiến thức về hồi quy, phân loại, phân cụm và mạng nơ-ron; đồng thời rèn luyện kỹ năng sử dụng Python và các thư viện hỗ trợ để giải quyết những bài toán dữ liệu.

---

**Repository:** [Machine_Learning_VLU](https://github.com/HungHIHI1603/Machine_Learning_VLU)  
**Course:** Machine Learning — Học Máy Và Ứng Dụng  
**University:** Văn Lang University (VLU)  
**Class:** 261_71ITAI41203_0102  
**Labs:** 01 – 09
