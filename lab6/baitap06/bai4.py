# -*- coding: utf-8 -*-
"""Bài tập 4: Dò lưới mịn quanh cặp C và gamma tốt nhất."""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Đọc bộ dữ liệu hai vầng trăng
df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

# Chia dữ liệu giống mục 5 của tài liệu
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Tạo lưới tham số mịn theo yêu cầu đề bài
luoi = {
    "C": [3, 10, 30],
    "gamma": [0.3, 1, 3]
}

# Dùng GridSearchCV với kiểm định chéo 5 phần
tim = GridSearchCV(SVC(kernel="rbf"), luoi, cv=5)
tim.fit(X_tr, y_tr)

# In điểm kiểm định chéo của từng cặp tham số
bang = pd.DataFrame(tim.cv_results_["params"])
bang["diem"] = tim.cv_results_["mean_test_score"]
print("DIEM KIEM DINH CHEO CUA CAC CAP C VA GAMMA")
print(bang.pivot(index="C", columns="gamma", values="diem").round(4))
print()

# In cặp tham số tốt nhất và accuracy trên tập kiểm tra
print("Cap tham so tot nhat:", tim.best_params_)
print(f"Diem kiem dinh cheo tot nhat: {tim.best_score_:.4f}")
print(f"Accuracy tren tap kiem tra: {tim.score(X_te, y_te):.4f}")
print()

print("Nhan xet: Luoi min tim duoc cap tot hon neu diem kiem dinh cheo tang.")
print("Accuracy tap kiem tra duoc so sanh voi muc 0.9200 cua muc 5.")
