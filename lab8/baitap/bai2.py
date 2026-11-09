# -*- coding: utf-8 -*-
from pathlib import Path
"""Bài tập 2: thêm tuổi, chuẩn hoá và chọn K bằng silhouette."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "khach_hang.csv")
cot = ["tuoi", "thu_nhap", "diem_chi_tieu"]
sc = StandardScaler()
Xs = sc.fit_transform(df[cot])
print("K   silhouette")
diem = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    diem[k] = silhouette_score(Xs, km.labels_)
    print(f"{k:2d}  {diem[k]:.4f}")
k_tot = max(diem, key=diem.get)
km = KMeans(n_clusters=k_tot, n_init=10, random_state=0).fit(Xs)
tam = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=cot)
tam["so_khach"] = pd.Series(km.labels_).value_counts().sort_index().values
print(f"\nK tot nhat khi them tuoi: {k_tot}")
print("Tam cum (don vi goc):")
print(tam.round(2).to_string(index=False))
print("So sanh: khi them tuoi, khoang cach thay doi nen tam cum va K tot nhat co the khac K=5 o muc 3.")
