# -*- coding: utf-8 -*-
from pathlib import Path
"""Bài tập 4: tự vẽ ROC của Logistic và Random Forest, lưu roc.png."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "sinh_vien_bo_hoc.csv")
X, y = df.drop(columns="bo_hoc"), df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y)
models = {
    "Logistic": make_pipeline(StandardScaler(), LogisticRegression()),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0)
}
plt.figure(figsize=(8, 6))
for name, model in models.items():
    model.fit(X_tr, y_tr)
    score = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, score)
    auc = roc_auc_score(y_te, score)
    plt.plot(fpr, tpr, label=f"{name}, AUC = {auc:.4f}")
    print(f"{name}: AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], "--", label="Du doan ngau nhien")
plt.xlabel("Ty le bao dong nham (FPR)")
plt.ylabel("Recall (TPR)")
plt.title("Duong ROC")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent.parent / "roc.png", dpi=150)
print("Da luu roc.png")
print("Nhan xet: duong nao gan goc tren trai hon va co AUC lon hon thi tot hon; quan sat giao diem tren do thi.")
