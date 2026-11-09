# -*- coding: utf-8 -*-
"""Bài tập 2: kiểm chứng mạng XOR thứ hai bằng numpy."""
import numpy as np

def relu(z):
    return np.maximum(0, z)

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
print("x1 x2 h1 h2 y")
for x1, x2 in X:
    h1 = relu(x1 - x2)
    h2 = relu(x2 - x1)
    y = h1 + h2
    print(f"{x1:.0f}  {x2:.0f}  {h1:.0f}  {h2:.0f}  {y:.0f}")
print("Nhan xet: hai dau vao bang nhau thi h1=h2=0, y=0; khac nhau thi mot ReLU bang 1, y=1.")
