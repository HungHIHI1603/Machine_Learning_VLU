# -*- coding: utf-8 -*-
"""Bài tập 3: Dò tham số C cho kernel đa thức bậc 2, 3 và 4."""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Đọc dữ liệu hai vầng trăng
df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

# Chia tập học và tập kiểm tra đúng theo tài liệu
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Thử lần lượt ba bậc đa thức
ket_qua = []

for bac in [2, 3, 4]:
    # Chuẩn hóa dữ liệu rồi mới đưa vào SVM đa thức
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel="poly", degree=bac)
    )

    # Dò bốn giá trị C bằng kiểm định chéo 5 phần
    luoi = {"svc__C": [0.1, 1, 10, 100]}
    tim = GridSearchCV(pipeline, luoi, cv=5)
    tim.fit(X_tr, y_tr)

    # Tính accuracy trên tập kiểm tra bằng mô hình tốt nhất
    accuracy = accuracy_score(y_te, tim.predict(X_te))

    ket_qua.append({
        "Bac": bac,
        "C tot nhat": tim.best_params_["svc__C"],
        "Diem kiem dinh cheo": tim.best_score_,
        "Accuracy tap kiem tra": accuracy
    })

    print(f"KERNEL DA THUC BAC {bac}")
    print("C tot nhat:", tim.best_params_["svc__C"])
    print(f"Diem kiem dinh cheo: {tim.best_score_:.4f}")
    print(f"Accuracy tap kiem tra: {accuracy:.4f}")
    print()

# In bảng tổng hợp để dễ so sánh
bang = pd.DataFrame(ket_qua)
print("BANG TONG HOP")
print(bang.to_string(index=False))

print()
print("Nhan xet: So sanh diem kiem dinh cheo va accuracy cua ba bac.")
print("Kernel RBF o muc 5 co accuracy tap kiem tra 0.9200.")
print("Neu accuracy cua kernel da thuc thap hon 0.9200 thi khong vuot RBF.")
