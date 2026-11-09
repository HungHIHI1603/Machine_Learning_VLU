# -*- coding: utf-8 -*-
from pathlib import Path
"""Bài tập 3: so sánh logistic, tanh và relu trên hai vầng trăng."""
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "hai_vang_trang.csv")
X, y = df[["x1", "x2"]], df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y)
for activation in ["logistic", "tanh", "relu"]:
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100,100), alpha=1.0,
                      activation=activation, solver="lbfgs",
                      max_iter=3000, random_state=0)
    ).fit(X_tr, y_tr)
    print(f"{activation:10s} tap hoc={accuracy_score(y_tr, model.predict(X_tr)):.4f},"
          f" tap kiem tra={accuracy_score(y_te, model.predict(X_te)):.4f}")
print("Nhan xet: so sanh accuracy tap kiem tra; ham co gia tri cao nhat la tot nhat tren lan chia nay.")
