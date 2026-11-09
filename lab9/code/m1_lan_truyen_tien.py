# -*- coding: utf-8 -*-
"""Bước 1: tự tính lan truyền tiến bằng numpy."""
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

# Phần 1: một nơ-ron với hàm sigmoid, chính là hồi quy logistic
w = np.array([0.8, -0.5, 0.3])
b = -0.2
x = np.array([2.0, 1.0, 3.0])
z = w @ x + b
print("Mot no-ron:")
print(f" z = {w[0]}*{x[0]} + ({w[1]})*{x[1]} + {w[2]}*{x[2]} + ({b}) = {z:.2f}")
print(f" a = sigmoid(z) = {sigmoid(z):.4f}")
print()
# Phần 2: mạng 2 đầu vào, 2 nơ-ron ẩn ReLU, 1 nơ-ron ra
W1 = np.array([[1.0, 1.0], [1.0, 1.0]])
b1 = np.array([0.0, -1.0])
W2 = np.array([1.0, -2.0])
b2 = 0.0
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
print("Mang giai XOR:")
print(" x1 x2     h1 h2     y")
for x in X:
    h = relu(x @ W1 + b1)
    y = h @ W2 + b2
    print(f" {x[0]:.0f}    {x[1]:.0f}    {h[0]:.0f}   {h[1]:.0f}    {y:.0f}")
print()
H = relu(X @ W1 + b1)
Y = H @ W2 + b2
print("Tinh ca bang mot lan:", Y)
