# -*- coding: utf-8 -*-
from pathlib import Path
"""Bài tập 5: cân bằng lớp cho Random Forest và SVC."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "sinh_vien_bo_hoc.csv")
X, y = df.drop(columns="bo_hoc"), df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y)
models = {
    "RF thuong": RandomForestClassifier(n_estimators=200, random_state=0),
    "RF balanced": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0),
    "SVC thuong": make_pipeline(StandardScaler(), SVC()),
    "SVC balanced": make_pipeline(StandardScaler(), SVC(class_weight="balanced"))
}
ket_qua = {}
for name, model in models.items():
    pred = model.fit(X_tr, y_tr).predict(X_te)
    scores = (precision_score(y_te, pred, zero_division=0),
              recall_score(y_te, pred, zero_division=0),
              f1_score(y_te, pred, zero_division=0))
    ket_qua[name] = scores
    print(f"{name:14s} precision={scores[0]:.4f}, recall={scores[1]:.4f}, F1={scores[2]:.4f}")
print("\nNhan xet: class_weight='balanced' thuong tang recall lop hiem, doi lai precision co the giam.")
