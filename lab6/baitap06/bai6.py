# -*- coding: utf-8 -*-
"""Bài tập 6: So sánh thời gian học SVC và LinearSVC trên dữ liệu lớn."""

import time
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.svm import SVC, LinearSVC

# Các kích thước dữ liệu cần thử theo đề bài
cac_co_du_lieu = [2000, 5000, 10000, 20000]

# Danh sách lưu kết quả thời gian chạy
ket_qua = []

for n in cac_co_du_lieu:
    # Sinh dữ liệu gồm n mẫu và 20 đặc trưng
    X, y = make_classification(
        n_samples=n,
        n_features=20,
        random_state=0
    )

    # Đo thời gian huấn luyện SVC
    mo_hinh_svc = SVC()
    bat_dau = time.perf_counter()
    mo_hinh_svc.fit(X, y)
    thoi_gian_svc = time.perf_counter() - bat_dau

    # Đo thời gian huấn luyện LinearSVC
    mo_hinh_linear = LinearSVC()
    bat_dau = time.perf_counter()
    mo_hinh_linear.fit(X, y)
    thoi_gian_linear = time.perf_counter() - bat_dau

    # Lưu kết quả của kích thước dữ liệu hiện tại
    ket_qua.append({
        "So mau": n,
        "Thoi gian SVC (giay)": thoi_gian_svc,
        "Thoi gian LinearSVC (giay)": thoi_gian_linear
    })

# Chuyển kết quả thành bảng và in ra màn hình
bang = pd.DataFrame(ket_qua)
print("BANG SO SANH THOI GIAN HUAN LUYEN")
print(bang.to_string(index=False, formatters={
    "Thoi gian SVC (giay)": "{:.6f}".format,
    "Thoi gian LinearSVC (giay)": "{:.6f}".format
}))

print()
print("SO SANH KHI TANG SO MAU")
for i in range(1, len(bang)):
    ti_le_svc = bang.loc[i, "Thoi gian SVC (giay)"] / bang.loc[i - 1, "Thoi gian SVC (giay)"]
    ti_le_linear = bang.loc[i, "Thoi gian LinearSVC (giay)"] / bang.loc[i - 1, "Thoi gian LinearSVC (giay)"]
    print(
        f"{int(bang.loc[i - 1, 'So mau'])} -> {int(bang.loc[i, 'So mau'])} mau: "
        f"SVC x{ti_le_svc:.2f}, LinearSVC x{ti_le_linear:.2f}"
    )

print()
print("Nhan xet: Thoi gian phu thuoc vao cau hinh may va du lieu.")
print("Thong thuong SVC co the cham hon ro khi so mau tang,")
print("con LinearSVC phu hop hon voi du lieu lon va bai toan tuyen tinh.")
