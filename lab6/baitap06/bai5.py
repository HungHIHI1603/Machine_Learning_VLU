# -*- coding: utf-8 -*-
"""Bài tập 5: Ma trận nhầm lẫn và lưu các ảnh chữ số dự đoán sai."""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Nạp bộ dữ liệu chữ số viết tay có sẵn trong scikit-learn
X, y = load_digits(return_X_y=True)

# Chia tập học và tập kiểm tra giống mục 6
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Huấn luyện SVM kernel RBF
svm = SVC(kernel="rbf")
svm.fit(X_tr, y_tr)

# Dự đoán nhãn cho tập kiểm tra
du_doan = svm.predict(X_te)

# In accuracy và ma trận nhầm lẫn 10 hàng, 10 cột
print(f"Accuracy SVM: {accuracy_score(y_te, du_doan):.4f}")
ma_tran = confusion_matrix(y_te, du_doan, labels=np.arange(10))
print("MA TRAN NHAM LAN (HANG = NHAN THAT, COT = NHAN DU DOAN)")
print(ma_tran)

# Tìm vị trí những ảnh dự đoán sai
cac_vi_tri_sai = np.flatnonzero(du_doan != y_te)
print()
print("CAC ANH DU DOAN SAI")
for i in cac_vi_tri_sai:
    print(f"Vi tri {i}: nhan that = {y_te[i]}, du doan = {du_doan[i]}")

# Tạo thư mục để lưu ảnh sai nếu thư mục chưa tồn tại
os.makedirs("baitap06/anh_du_doan_sai", exist_ok=True)

# Vẽ và lưu riêng từng ảnh bị dự đoán sai
for i in cac_vi_tri_sai:
    plt.figure(figsize=(3, 3))
    plt.imshow(X_te[i].reshape(8, 8), cmap="gray_r")
    plt.title(f"That: {y_te[i]} - Doan: {du_doan[i]}")
    plt.axis("off")
    ten_anh = f"baitap06/anh_du_doan_sai/anh_{i}_that_{y_te[i]}_doan_{du_doan[i]}.png"
    plt.savefig(ten_anh, bbox_inches="tight")
    plt.close()

print()
print(f"Da luu {len(cac_vi_tri_sai)} anh sai trong baitap06/anh_du_doan_sai")

print("Nhan xet: Cac chu so viet tay co net gan giong nhau hoac anh khong ro")
print("co the lam SVM nham lan. Hay xem cac anh da luu de nhan xet tung truong hop.")
