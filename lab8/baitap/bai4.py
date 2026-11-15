# -*- coding: utf-8 -*-
"""Bài tập 4: lưu ảnh gốc, ảnh 16 màu và thử ảnh cá nhân nếu có."""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def nen_anh(anh, ten_tien_to):
    # Bỏ kênh alpha nếu ảnh PNG có 4 kênh.
    if anh.ndim == 2:
        anh = np.stack([anh, anh, anh], axis=-1)
    if anh.shape[2] == 4:
        anh = anh[:, :, :3]
    anh = np.asarray(anh)
    if np.issubdtype(anh.dtype, np.integer):
        anh_float = anh.astype(float) / 255.0
    else:
        anh_float = np.clip(anh.astype(float), 0, 1)
    h, w, _ = anh_float.shape
    points = anh_float.reshape(-1, 3)
    rng = np.random.default_rng(0)
    sample = points[rng.choice(len(points), min(5000, len(points)), replace=False)]
    km = KMeans(n_clusters=16, n_init=4, random_state=0).fit(sample)
    compressed = km.cluster_centers_[km.predict(points)].reshape(h, w, 3)
    plt.imsave(Path(__file__).resolve().parent.parent / f"{ten_tien_to}_goc.png", anh_float)
    plt.imsave(Path(__file__).resolve().parent.parent / f"{ten_tien_to}_16_mau.png", compressed)
    print(f"{ten_tien_to}: da luu anh goc va anh 16 mau, kich thuoc {h}x{w}")
    print("Dung luong byte:", os.path.getsize(Path(__file__).resolve().parent.parent / f"{ten_tien_to}_goc.png"),
          "->", os.path.getsize(Path(__file__).resolve().parent.parent / f"{ten_tien_to}_16_mau.png"))

duong_dan_anh_mau = Path(__file__).resolve().parent.parent / "anh_mau_goc.png"
anh = plt.imread(duong_dan_anh_mau)
nen_anh(anh, "anh_mau")
# Nếu đặt ảnh cá nhân tại data/anh_cua_em.png thì chương trình tự xử lý.
duong_dan = Path(__file__).resolve().parent.parent / "data" / "anh_cua_em.png"
if duong_dan.exists():
    nen_anh(plt.imread(duong_dan), "anh_cua_em")
else:
    print("Chua co data/anh_cua_em.png; dat anh cua ban vao dung duong dan nay de lam phan anh ca nhan.")
