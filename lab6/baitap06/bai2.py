# -*- coding: utf-8 -*-
"""Bài tập 2: Thêm từng điểm mới và quan sát sự thay đổi ranh giới."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# Sinh lại 24 điểm ban đầu giống mục 2
X, y = make_blobs(
    n_samples=24,
    centers=[[2, 2], [6, 6]],
    cluster_std=0.9,
    random_state=0
)
y = 1 - y  # Lớp 1 là cụm dưới bên trái

# Huấn luyện và in hệ số của mô hình ban đầu
svm_goc = SVC(kernel="linear", C=1000)
svm_goc.fit(X, y)

print("MO HINH BAN DAU")
print(f"w1 = {svm_goc.coef_[0][0]:.3f}")
print(f"w2 = {svm_goc.coef_[0][1]:.3f}")
print(f"b  = {svm_goc.intercept_[0]:.3f}")
print("Chi so vecto ho tro:", svm_goc.support_)
print()

# Trường hợp 1: thêm điểm lớp 1 tại (-1, -1)
diem_1 = np.array([[-1.0, -1.0]])
X_them_1 = np.vstack((X, diem_1))
y_them_1 = np.append(y, 1)

svm_1 = SVC(kernel="linear", C=1000)
svm_1.fit(X_them_1, y_them_1)

print("TRUONG HOP 1: THEM DIEM LOP 1 TAI (-1, -1)")
print(f"w1 = {svm_1.coef_[0][0]:.3f}")
print(f"w2 = {svm_1.coef_[0][1]:.3f}")
print(f"b  = {svm_1.intercept_[0]:.3f}")
print("Chi so vecto ho tro:", svm_1.support_)
print()

# Trường hợp 2: thay điểm mới bằng điểm lớp 1 tại (4.5, 4)
diem_2 = np.array([[4.5, 4.0]])
X_them_2 = np.vstack((X, diem_2))
y_them_2 = np.append(y, 1)

svm_2 = SVC(kernel="linear", C=1000)
svm_2.fit(X_them_2, y_them_2)

print("TRUONG HOP 2: THEM DIEM LOP 1 TAI (4.5, 4)")
print(f"w1 = {svm_2.coef_[0][0]:.3f}")
print(f"w2 = {svm_2.coef_[0][1]:.3f}")
print(f"b  = {svm_2.intercept_[0]:.3f}")
print("Chi so vecto ho tro:", svm_2.support_)
print()

# So sánh các hệ số để xác định ranh giới có thay đổi hay không
print("KET LUAN")
print("Diem (-1, -1) khong lam thay doi ranh gioi vi khong tro thanh vecto ho tro.")
print("Diem (4.5, 4) lam thay doi ranh gioi vi no nam gan/chen vao vung le")
print("va tro thanh vecto ho tro, nen tac dong den duong phan chia.")
