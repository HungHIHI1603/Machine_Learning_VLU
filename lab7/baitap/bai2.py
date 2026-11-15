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
from pathlib import Path
import pandas as pd


# Hàm tìm và đọc file CSV trong thư mục data
def doc_csv(ten_file):
    # Lấy đường dẫn của file Python hiện tại
    file_hien_tai = Path(__file__).resolve()

    # Duyệt qua thư mục hiện tại và các thư mục cha
    for thu_muc in file_hien_tai.parents:

        # Xác định thư mục chứa dữ liệu
        thu_muc_data = thu_muc / "data"

        # Kiểm tra thư mục data có tồn tại không
        if thu_muc_data.exists():

            # Tìm file đúng tên
            duong_dan = thu_muc_data / ten_file

            # Nếu tìm thấy thì đọc dữ liệu
            if duong_dan.exists():
                print("Đã đọc dữ liệu:", duong_dan)
                return pd.read_csv(duong_dan)

            # Tìm cả file có tên kèm số như gia_nha(3).csv
            ten_goc = Path(ten_file).stem.lower()

            for file in thu_muc_data.glob("*.csv"):
                ten_thuc_te = file.stem.lower()

                if (
                    ten_thuc_te == ten_goc
                    or ten_thuc_te.startswith(ten_goc + "(")
                ):
                    print("Đã đọc dữ liệu:", file)
                    return pd.read_csv(file)

    # Báo lỗi nếu không tìm thấy file
    raise FileNotFoundError(
        f"Không tìm thấy {ten_file}. "
        "Hãy kiểm tra thư mục data."
    )