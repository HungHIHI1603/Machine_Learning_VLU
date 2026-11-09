
# -*- coding: utf-8 -*-

"""Bước 2: lề mềm, tham số C, và vì sao SVM phải chuẩn hoá."""

# Import các thư viện cần thiết
import numpy as np
import pandas as pd

# Import thư viện xử lý đường dẫn
from pathlib import Path

# Import các hàm đánh giá và chia dữ liệu
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Import Pipeline và chuẩn hóa dữ liệu
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Import mô hình SVM
from sklearn.svm import SVC


# ============================================================
# 1. XÁC ĐỊNH ĐƯỜNG DẪN THƯ MỤC DATA
# ============================================================

# Lấy thư mục chứa file Python hiện tại
THU_MUC_CODE = Path(__file__).resolve().parent

# Lùi ra một cấp để đến thư mục baitap06_svm
THU_MUC_GOC = THU_MUC_CODE.parent

# Xác định thư mục chứa dữ liệu CSV
THU_MUC_DATA = THU_MUC_GOC / "data"


# ============================================================
# 2. ĐỌC DỮ LIỆU SINH VIÊN
# ============================================================

# Đọc file sinh_vien.csv từ thư mục data
df = pd.read_csv(THU_MUC_DATA / "sinh_vien.csv")

# Lấy hai cột đặc trưng đầu vào
X = df[["gio_on", "diem_giua_ky"]]

# Lấy cột nhãn đầu ra
y = df["qua_mon"]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)


# ============================================================
# 3. KHẢO SÁT ẢNH HƯỞNG CỦA THAM SỐ C
# ============================================================

# In số lượng mẫu huấn luyện
print("Tap hoc co", len(X_tr), "ban")

# In tiêu đề bảng kết quả
print("    C    so vecto HT   do rong le    tap hoc   tap kiem tra")

# Thử nghiệm các giá trị C khác nhau
for C in [0.01, 0.1, 1, 10, 100]:

    # Tạo Pipeline gồm chuẩn hóa và mô hình SVM tuyến tính
    mh = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=C)
    )

    # Huấn luyện mô hình
    mh.fit(X_tr, y_tr)

    # Lấy mô hình SVM ở bước cuối của Pipeline
    svm = mh[-1]

    # Tính độ rộng lề bằng công thức 2 / ||w||
    le = 2 / np.linalg.norm(svm.coef_[0])

    # Tính độ chính xác trên tập huấn luyện
    a_hoc = accuracy_score(y_tr, mh.predict(X_tr))

    # Tính độ chính xác trên tập kiểm tra
    a_kt = accuracy_score(y_te, mh.predict(X_te))

    # In kết quả của từng giá trị C
    print(
        f" {C:6}    {len(svm.support_):6d}        {le:6.2f}"
        f"      {a_hoc:.4f}     {a_kt:.4f}"
    )

print()


# ============================================================
# 4. THỬ NGHIỆM VỚI BỘ SINH VIÊN MỞ RỘNG
# ============================================================

# Đọc dữ liệu sinh viên mở rộng
df3 = pd.read_csv(THU_MUC_DATA / "sinh_vien_mo_rong.csv")

# Lấy ba cột đặc trưng đầu vào
X3 = df3[["gio_on", "diem_giua_ky", "chi_tieu"]]

# Lấy cột nhãn đầu ra
y3 = df3["qua_mon"]

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X3_tr, X3_te, y3_tr, y3_te = train_test_split(
    X3,
    y3,
    test_size=0.25,
    random_state=17,
    stratify=y3
)

# Tạo và huấn luyện SVM mặc định, không chuẩn hóa
tho = SVC().fit(X3_tr, y3_tr)

# Tạo và huấn luyện SVM có chuẩn hóa dữ liệu
chuan = make_pipeline(
    StandardScaler(),
    SVC()
).fit(X3_tr, y3_tr)


# ============================================================
# 5. SO SÁNH KẾT QUẢ
# ============================================================

# In tiêu đề kết quả
print("Ba cot, kernel RBF mac dinh")

# In tỷ lệ sinh viên qua môn trong tập kiểm tra
print(f" Doan bua ca lop qua : {y3_te.mean():.4f}")

# In độ chính xác khi không chuẩn hóa
print(
    f" Khong chuan hoa      : "
    f"{accuracy_score(y3_te, tho.predict(X3_te)):.4f}"
)

# In độ chính xác khi có chuẩn hóa
print(
    f" Co chuan hoa         : "
    f"{accuracy_score(y3_te, chuan.predict(X3_te)):.4f}"
)
