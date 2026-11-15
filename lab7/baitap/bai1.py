# -*- coding: utf-8 -*-
from pathlib import Path
"""Bài tập 1: tự viết hàm đánh giá chỉ dùng numpy."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def danh_gia(y_that, y_doan):
    """Tự tính accuracy, precision, recall, F1; mẫu số 0 thì trả về 0."""
    y_that = np.asarray(y_that)
    y_doan = np.asarray(y_doan)
    tp = np.sum((y_that == 1) & (y_doan == 1))
    fp = np.sum((y_that == 0) & (y_doan == 1))
    fn = np.sum((y_that == 1) & (y_doan == 0))
    tn = np.sum((y_that == 0) & (y_doan == 0))
    acc = (tp + tn) / len(y_that) if len(y_that) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "sinh_vien_bo_hoc.csv")
X, y = df.drop(columns="bo_hoc"), df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y)
models = {
    "Logistic": make_pipeline(StandardScaler(), LogisticRegression()),
    "Mo hinh luoi": DummyClassifier(strategy="most_frequent")
}
for name, model in models.items():
    pred = model.fit(X_tr, y_tr).predict(X_te)
    print("\n", name)
    print("Tu viet:", danh_gia(y_te, pred))
    print("sklearn:", {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred, zero_division=0)
    })
print("\nNhan xet: mo hinh luoi khong du doan mau duong nao, precision = 0 theo quy uoc zero_division=0.")
