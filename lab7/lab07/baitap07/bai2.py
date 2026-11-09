# -*- coding: utf-8 -*-
"""Bài tập 2: classification_report cho SVM nhận dạng chữ số."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y)
svm = SVC(kernel="rbf").fit(X_tr, y_tr)
pred = svm.predict(X_te)
print(classification_report(y_te, pred, digits=4))
report = classification_report(y_te, pred, output_dict=True, zero_division=0)
classes = [str(i) for i in range(10)]
recalls = {c: report[c]["recall"] for c in classes}
precisions = {c: report[c]["precision"] for c in classes}
print("Recall thap nhat:", min(recalls, key=recalls.get), min(recalls.values()))
print("Precision thap nhat:", min(precisions, key=precisions.get), min(precisions.values()))
print("Nhan xet: digits co so mau moi lop gan bang nhau, nen macro avg va weighted avg gan nhau.")
