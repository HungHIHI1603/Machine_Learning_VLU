# -*- coding: utf-8 -*-
from pathlib import Path
"""Bài tập 6: MAE, RMSE, R2 trước và sau một dự đoán bị cộng thêm 5 tỷ."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "gia_nha.csv")
X = df[["dien_tich", "so_phong", "tuoi_nha"]]
y = df["gia"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17)
model = LinearRegression().fit(X_tr, y_tr)
pred = model.predict(X_te)
def in_chi_so(ten, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{ten}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
in_chi_so("Ban dau", y_te, pred)
pred_hong = pred.copy()
pred_hong[0] += 5.0  # giá trong CSV tính bằng tỷ đồng
in_chi_so("Sau khi cong 5 ty vao mot can", y_te, pred_hong)
print("Nhan xet: RMSE thay doi manh hon MAE vi binh phuong sai so lon truoc khi lay can.")
