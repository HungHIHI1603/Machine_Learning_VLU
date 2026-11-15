# -*- coding: utf-8 -*-
from pathlib import Path
"""Bài tập 3: so sánh K-Means init=random và k-means++ trong 50 lần chạy."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "khach_hang.csv")
X = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])
for init in ["random", "k-means++"]:
    inertia = []
    for rs in range(50):
        km = KMeans(n_clusters=5, init=init, n_init=1, random_state=rs).fit(X)
        inertia.append(km.inertia_)
    count = sum(round(v, 1) == 23.0 for v in inertia)
    print(f"{init:10s}: dat inertia 23.0 trong {count}/50 lan")
    print("  min, max inertia:", round(min(inertia), 3), round(max(inertia), 3))
print("Nhan xet: cach nao dat 23.0 nhieu lan hon va co bien do inertia nho hon thi dang tin cay hon.")
