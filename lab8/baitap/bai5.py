# -*- coding: utf-8 -*-
from pathlib import Path
"""Bài tập 5: DBSCAN trên bộ hai vầng trăng."""
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "hai_vang_trang.csv")
X = df[["x1", "x2"]].to_numpy()
y = df["nhan"].to_numpy()
for e in [0.2, 0.3, 0.4]:
    nhan = DBSCAN(eps=e, min_samples=5).fit_predict(X)
    so_cum = len(set(nhan)) - (1 if -1 in nhan else 0)
    so_nhieu = int((nhan == -1).sum())
    ari = adjusted_rand_score(y, nhan)
    print(f"eps={e}: so cum={so_cum}, so diem nhieu={so_nhieu}, ARI={ari:.4f}")
print("Nhan xet: eps nho tao nhieu diem nhieu/tach cum; eps lon co the gop hai vong thanh mot cum.")
