
# -*- coding: utf-8 -*-

# ============================================================
# BAI TAP 4: SO SANH TOC DO HOC CUA MANG NEURAL NETWORK
# ============================================================

# Import thu vien xu ly duong dan
from pathlib import Path

# Import thu vien ve bieu do
import matplotlib.pyplot as plt

# Import bo du lieu chu so viet tay
from sklearn.datasets import load_digits

# Import ham danh gia do chinh xac
from sklearn.metrics import accuracy_score

# Import ham chia du lieu
from sklearn.model_selection import train_test_split

# Import mo hinh mang noron MLP
from sklearn.neural_network import MLPClassifier

# Import Pipeline va chuan hoa du lieu
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. TAI DU LIEU
# ============================================================

# Tai bo du lieu chu so viet tay
X, y = load_digits(return_X_y=True)

# Chia du lieu thanh tap huan luyen va tap kiem tra
X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)


# ============================================================
# 2. KHAI BAO CAC TOC DO HOC
# ============================================================

# Danh sach cac gia tri learning rate can so sanh
rates = [0.0001, 0.001, 0.01, 0.1]

# Tao khung bieu do
plt.figure(figsize=(9, 6))


# ============================================================
# 3. HUAN LUYEN VA SO SANH MO HINH
# ============================================================

# Tao danh sach luu ket qua
ket_qua = []

# Lap qua tung gia tri learning rate
for rate in rates:

    # Tao Pipeline gom chuan hoa va mo hinh MLP
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(100,),
            learning_rate_init=rate,
            max_iter=500,
            random_state=0,
            solver="adam",
            early_stopping=False
        )
    )

    # Huan luyen mo hinh
    model.fit(X_tr, y_tr)

    # Lay mo hinh MLP tu Pipeline
    mlp = model[-1]

    # Du doan tren tap kiem tra
    y_pred = model.predict(X_te)

    # Tinh do chinh xac
    acc = accuracy_score(y_te, y_pred)

    # Luu ket qua
    ket_qua.append((rate, mlp.n_iter_, mlp.loss_, acc))

    # In ket qua
    print(
        f"learning_rate={rate}: "
        f"n_iter={mlp.n_iter_}, "
        f"loss cuoi={mlp.loss_:.6f}, "
        f"accuracy={acc:.4f}"
    )

    # Ve duong loss theo tung vong hoc
    plt.plot(
        range(1, len(mlp.loss_curve_) + 1),
        mlp.loss_curve_,
        label=f"learning_rate={rate}"
    )


# ============================================================
# 4. VE BIEU DO SO SANH
# ============================================================

# Dat ten truc ngang
plt.xlabel("Vong hoc")

# Dat ten truc doc
plt.ylabel("Loss")

# Dat tieu de bieu do
plt.title("So sanh toc do hoc")

# Hien thi chu thich
plt.legend()

# Hien thi luoi
plt.grid(True)

# Can chinh bo cuc bieu do
plt.tight_layout()


# ============================================================
# 5. LUU BIEU DO
# ============================================================

# Xac dinh thu muc chua file Python dang chay
thu_muc_hien_tai = Path(__file__).resolve().parent

# Tao thu muc neu chua ton tai
thu_muc_hien_tai.mkdir(parents=True, exist_ok=True)

# Tao duong dan luu bieu do
duong_dan_anh = thu_muc_hien_tai / "toc_do_hoc.png"

# Luu bieu do
plt.savefig(duong_dan_anh, dpi=150)

# Thong bao luu thanh cong
print("\nDa luu bieu do tai:", duong_dan_anh)


# ============================================================
# 6. NHAN XET
# ============================================================

# Tim learning rate co accuracy cao nhat
ket_qua_tot_nhat = max(ket_qua, key=lambda x: x[3])

# In ket qua tot nhat
print("\nKET QUA TOT NHAT:")
print("Learning rate:", ket_qua_tot_nhat[0])
print("So vong hoc:", ket_qua_tot_nhat[1])
print("Loss cuoi:", ket_qua_tot_nhat[2])
print("Accuracy:", ket_qua_tot_nhat[3])

# Dua ra nhan xet
print("\nNHAN XET:")
print(
    "Toc do hoc qua nho thi hoi tu cham; "
    "qua lon co the dao dong hoac khong hoi tu on dinh."
)

# Hien thi bieu do
plt.show()

# Thong bao hoan thanh
print("\nHOAN THANH BAI TAP 4!")
