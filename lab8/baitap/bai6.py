# -*- coding: utf-8 -*-
"""Bài tập 6: K-Means K=10 trên digits, sau đó mới đối chiếu nhãn thật."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
# Không đưa y vào quá trình phân cụm.
Xs = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(Xs)
bang = pd.crosstab(pd.Series(y, name="nhan_that"),
                   pd.Series(km.labels_, name="cum"))
print("Bang nhan that x cum:")
print(bang)
print("\nSo chu so trong tung cum:")
print(bang.idxmax(axis=0))
print("Nhan xet: cot co mot hang ap dao la chu so duoc gom gon; cot co nhieu hang lon la cac chu so hay bi gom chung.")
