# -*- coding: utf-8 -*-
"""Bước 5: nén màu một bức ảnh bằng K-Means."""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

duong_dan_anh_mau = Path(__file__).resolve().parent.parent / "anh_mau_goc.png"
anh = plt.imread(duong_dan_anh_mau)
# Loại bỏ kênh alpha nếu ảnh PNG có 4 kênh màu.
if anh.ndim == 3 and anh.shape[2] == 4:
    anh = anh[:, :, :3]
# Đưa ảnh về kiểu số thực trong khoảng 0 đến 1.
anh = anh.astype(float)
if anh.max() > 1:
    anh = anh / 255.0
cao, rong, _ = anh.shape
diem = anh.reshape(-1, 3)
so_mau = len(np.unique((anh * 255).astype(np.uint8).reshape(-1, 3), axis=0))
print(f"Anh {cao} x {rong}, co {cao * rong} diem anh, {so_mau} mau")
print()
rng = np.random.default_rng(0)
mau = diem[rng.choice(len(diem), 5000, replace=False)]
print(" K    so bit moi diem   nho hon   sai lech trung binh")
for k in [4, 16, 64]:
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(mau)
    nhan = km.predict(diem)
    moi = km.cluster_centers_[nhan]
    bit = int(np.ceil(np.log2(k)))
    sai = np.abs(moi - diem).mean() * 255
    print(f" {k:2d}         {bit}                  {24 / bit:4.1f} lan"
          f"      {sai:5.2f} muc sang")
