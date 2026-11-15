# -*- coding: utf-8 -*-
from pathlib import Path
"""Bài tập 3: chọn ngưỡng lớn nhất để recall kiểm định chéo >= 0.8."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "sinh_vien_bo_hoc.csv")
X, y = df.drop(columns="bo_hoc"), df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y)
model = make_pipeline(StandardScaler(), LogisticRegression())
# Xác suất out-of-fold trên tập học, không nhìn tập kiểm tra để chọn ngưỡng.
prob_cv = cross_val_predict(model, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
thresholds = np.unique(prob_cv)
valid = [(t, recall_score(y_tr, prob_cv >= t, zero_division=0))
         for t in thresholds]
valid = [(t, r) for t, r in valid if r >= 0.80]
if not valid:
    threshold = 0.0
    print("Khong tim thay nguong hop le; dung nguong 0.0 de dam bao recall.")
else:
    threshold = max(t for t, r in valid)
print(f"Nguong lon nhat co recall CV >= 0.8: {threshold:.6f}")
model.fit(X_tr, y_tr)
pred = (model.predict_proba(X_te)[:, 1] >= threshold).astype(int)
print(f"Recall tap kiem tra: {recall_score(y_te, pred, zero_division=0):.4f}")
print(f"Precision tap kiem tra: {precision_score(y_te, pred, zero_division=0):.4f}")
print("So ban bi bao:", int(pred.sum()), "/", len(pred))
print("Nhan xet: recall tren tap kiem tra co the khong dat 0.8 do du lieu moi; khong duoc chinh nguong theo tap kiem tra.")
