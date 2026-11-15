# -*- coding: utf-8 -*-
from pathlib import Path
"""Bài tập 1: tự tính silhouette cho từng khách hàng bằng numpy."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def silhouette_mot_diem(X, nhan, i):
    """Tính a, b, s cho một điểm theo định nghĩa silhouette."""
    X = np.asarray(X, dtype=float)
    nhan = np.asarray(nhan)
    khoang_cach = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
    cung_cum = (nhan == nhan[i])
    cung_cum[i] = False
    a = khoang_cach[cung_cum].mean() if cung_cum.any() else 0.0
    cac_cum = [c for c in np.unique(nhan) if c != nhan[i]]
    b = min(khoang_cach[nhan == c].mean() for c in cac_cum)
    s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return a, b, s

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "khach_hang.csv")
X = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])
km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X)
values = np.array([silhouette_mot_diem(X, km.labels_, i)[2] for i in range(len(X))])
print("Silhouette tu viet trung binh:", values.mean())
print("silhouette_score:", silhouette_score(X, km.labels_))
i_min = int(np.argmin(values))
print(f"Khach co silhouette nho nhat: dong {i_min}, s={values[i_min]:.4f}")
print("Thong tin khach:", df.iloc[i_min].to_dict())
print("s am/gan 0 cho thay diem gan cum khac hon hoac nam gan bien giua cac cum.")
