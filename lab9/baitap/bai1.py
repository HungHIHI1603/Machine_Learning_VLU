# -*- coding: utf-8 -*-
from pathlib import Path
"""Bài tập 1: tự lan truyền tiến bằng trọng số MLP đã học trên XOR."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def relu(z):
    return np.maximum(0, z)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "xor.csv")
X, y = df[["x1", "x2"]], df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y)
mlp = MLPClassifier(hidden_layer_sizes=(8,), solver="lbfgs",
                    max_iter=5000, random_state=0).fit(X_tr, y_tr)
W1, W2 = mlp.coefs_
b1, b2 = mlp.intercepts_
print("5 diem dau: xac suat tu tinh va predict_proba")
for x in X_te.iloc[:5].to_numpy():
    h = relu(x @ W1 + b1)
    z_out = h @ W2 + b2
    p_tu_tinh = sigmoid(z_out[0])
    p_sklearn = mlp.predict_proba(x.reshape(1, -1))[0, 1]
    print(f"x={x}, tu tinh={p_tu_tinh:.10f}, sklearn={p_sklearn:.10f},"
          f" chenhlech={abs(p_tu_tinh-p_sklearn):.2e}")
