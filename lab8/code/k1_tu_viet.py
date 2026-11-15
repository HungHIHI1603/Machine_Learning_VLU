# -*- coding: utf-8 -*-
"""Bước 1: tự viết K-Means bằng numpy, chạy trên bảy điểm của slide."""
import numpy as np
from sklearn.cluster import KMeans

P = np.array([[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0],
              [3.5, 5.0], [4.5, 5.0], [3.5, 4.5]])
TEN = ["A", "B", "C", "D", "E", "F", "G"]

def k_means(X, tam, so_vong_toi_da=10):
    """K-Means đơn giản: gán điểm vào tâm gần nhất rồi dời tâm."""
    for vong in range(1, so_vong_toi_da + 1):
        kc = np.sqrt(((X[:, None, :] - tam[None, :, :]) ** 2).sum(axis=2))
        nhan = kc.argmin(axis=1)
        print(f"Vong {vong}: tam = {np.round(tam, 2).tolist()}")
        for k in range(len(tam)):
            ds = [TEN[i] for i in range(len(X)) if nhan[i] == k]
            print(f" cum {k + 1}: {', '.join(ds)}")
        tam_moi = np.array([X[nhan == k].mean(axis=0) for k in range(len(tam))])
        if np.allclose(tam_moi, tam):
            print(" Tam khong doi nua, dung.")
            return tam, nhan
        tam = tam_moi
    return tam, nhan

# Khởi tạo cố ý tệ: hai tâm là A và B, nằm sát nhau
tam, nhan = k_means(P, P[[0, 1]].copy())
print()
inertia = ((P - tam[nhan]) ** 2).sum()
print(f"Inertia tu tinh: {inertia:.4f}")
km = KMeans(n_clusters=2, init=P[[0, 1]], n_init=1, random_state=0).fit(P)
print(f"Inertia cua scikit-learn: {km.inertia_:.4f}")
print("Tam cua scikit-learn:", np.round(km.cluster_centers_, 2).tolist())
