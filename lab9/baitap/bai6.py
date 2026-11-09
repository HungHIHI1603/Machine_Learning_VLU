# -*- coding: utf-8 -*-
from pathlib import Path
"""Bài tập 6: MLPRegressor so với hồi quy tuyến tính trên dữ liệu giá nhà."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "gia_nha.csv")
X = df[["dien_tich", "so_phong", "tuoi_nha"]]
y = df["gia"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17)
linear = LinearRegression().fit(X_tr, y_tr)
mlp = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000, random_state=0)
).fit(X_tr, y_tr)
pred_linear = linear.predict(X_te)
pred_mlp = mlp.predict(X_te)
print(f"MAE hoi quy tuyen tinh: {mean_absolute_error(y_te, pred_linear):.4f}")
print(f"MAE MLPRegressor:       {mean_absolute_error(y_te, pred_mlp):.4f}")
# Giữ các biến còn lại ở giá trị trung vị khi dự đoán nhà 250 m2.
moi = pd.DataFrame({
    "dien_tich": [250],
    "so_phong": [df["so_phong"].median()],
    "tuoi_nha": [df["tuoi_nha"].median()]
})
print("Du doan gia nha 250 m2 (cac cot khac o trung vi):")
print(f" Linear: {linear.predict(moi)[0]:.4f} ty")
print(f" MLP:    {mlp.predict(moi)[0]:.4f} ty")
print("Nhan xet: mang MLP khong co rang buoc tuyen tinh nen ngoai pham vi du lieu co the ngoai suy kem hon duong thang.")
