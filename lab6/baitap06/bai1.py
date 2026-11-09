# -*- coding: utf-8 -*-
"""Bài tập 1: Tính tay khoảng cách từ điểm đến ranh giới SVM."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# Sinh lại đúng 24 điểm như mục 2 của tài liệu
X, y = make_blobs(
    n_samples=24,
    centers=[[2, 2], [6, 6]],
    cluster_std=0.9,
    random_state=0
)
y = 1 - y  # Đổi nhãn để lớp 1 là cụm phía dưới bên trái

# Huấn luyện SVM tuyến tính với C lớn như tài liệu
svm = SVC(kernel="linear", C=1000)
svm.fit(X, y)

# Lấy các hệ số w1, w2 và b từ mô hình đã học
w1 = svm.coef_[0][0]
w2 = svm.coef_[0][1]
b = svm.intercept_[0]

# Tính độ dài vector w để dùng trong công thức khoảng cách
do_dai_w = np.sqrt(w1 ** 2 + w2 ** 2)

print("CAC HE SO CUA DUONG RANH GIOI")
print(f"w1 = {w1:.3f}, w2 = {w2:.3f}, b = {b:.3f}")
print(f"Do rong le = {2 / do_dai_w:.4f}")
print()

# Hai điểm mới cần tự tính f(x) bằng phép nhân và phép cộng
cac_diem = [(3.0, 3.0), (5.0, 4.5)]

print("KHOANG CACH CUA HAI DIEM MOI")
for x1, x2 in cac_diem:
    f_x = w1 * x1 + w2 * x2 + b  # Không dùng decision_function
    khoang_cach = abs(f_x) / do_dai_w
    print(f"Diem ({x1}, {x2}): f(x) = {f_x:.4f}")
    print(f"Khoang cach den ranh gioi = {abs(f_x):.4f} / {do_dai_w:.4f} = {khoang_cach:.4f}")
print()

# Lấy chỉ số của ba vectơ hỗ trợ do SVM xác định
print("KHOANG CACH CUA BA VECTO HO TRO")
for i in svm.support_:
    x1, x2 = X[i]
    f_x = w1 * x1 + w2 * x2 + b  # Tự tính bằng công thức f(x)
    khoang_cach = abs(f_x) / do_dai_w
    print(
        f"Diem {i}: ({x1:.4f}, {x2:.4f}), "
        f"f(x) = {f_x:.4f}, khoang cach = {khoang_cach:.4f}"
    )

print()
print(f"Một nửa độ rộng lề = {1 / do_dai_w:.4f}")
print("Giai thich: Vecto ho tro nam tren mep le f(x) = 1 hoac f(x) = -1.")
print("Vi vay khoang cach tu vecto ho tro den ranh gioi f(x) = 0 bang 1/||w||,")
print("dung bang mot nua do rong le 2/||w||.")
