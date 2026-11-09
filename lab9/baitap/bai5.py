# -*- coding: utf-8 -*-
"""Bài tập 5: GridSearchCV cho MLP trên bộ digits."""
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y)
pipe = make_pipeline(StandardScaler(),
    MLPClassifier(max_iter=1000, solver="adam", random_state=0,
                  early_stopping=True, n_iter_no_change=15))
grid = {
    "mlpclassifier__hidden_layer_sizes": [(10,), (50,), (100,), (50,50)],
    "mlpclassifier__alpha": [0.0001, 0.01, 1]
}
search = GridSearchCV(pipe, grid, cv=3, n_jobs=-1, scoring="accuracy")
search.fit(X_tr, y_tr)
print("Tham so tot nhat:", search.best_params_)
print(f"Accuracy CV tot nhat: {search.best_score_:.4f}")
mlp_acc = accuracy_score(y_te, search.predict(X_te))
svm = SVC().fit(X_tr, y_tr)
svm_acc = accuracy_score(y_te, svm.predict(X_te))
print(f"MLP accuracy tap kiem tra: {mlp_acc:.4f}")
print(f"SVM accuracy tap kiem tra: {svm_acc:.4f}")
print("MLP co vuot SVM khong?", mlp_acc > svm_acc)
