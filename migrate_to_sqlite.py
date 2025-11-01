# migrate_to_sqlite.py
import os, sqlite3, pyodbc, pandas as pd

# 1) Kết nối SQL Server local (chuỗi của bạn)
mssql = pyodbc.connect(
    "DRIVER={SQL Server};SERVER=HKT;DATABASE=CVD_App;UID=sa;PWD=123"
)

# 2) Đọc dữ liệu các bảng chính
nguoidung = pd.read_sql("SELECT * FROM NguoiDung", mssql)
chandoan  = pd.read_sql("SELECT * FROM ChanDoan",  mssql)
try:
    tinnhanai = pd.read_sql("SELECT * FROM TinNhanAI", mssql)
except:
    tinnhanai = pd.DataFrame(columns=["ID","BenhNhanID","NoiDung","PhanHoi","ThoiGian"])

mssql.close()

# 3) Tạo file SQLite
sqlite_path = os.path.join(os.path.dirname(__file__), "cvd_app.db")
if os.path.exists(sqlite_path):
    os.remove(sqlite_path)
sq = sqlite3.connect(sqlite_path)

# 4) Schema tối thiểu (tương thích cột)
sq.executescript("""
CREATE TABLE NguoiDung (
    ID INTEGER PRIMARY KEY,
    HoTen TEXT,
    Email TEXT,
    MatKhau TEXT,
    Role TEXT,
    NgaySinh TEXT,
    GioiTinh TEXT,
    DienThoai TEXT,
    DiaChi TEXT,
    NgayTao TEXT
);
CREATE TABLE ChanDoan (
    ID INTEGER PRIMARY KEY,
    BenhNhanID INTEGER,
    BacSiID INTEGER,
    NgayChanDoan TEXT,
    BMI REAL,
    HuyetApTamThu REAL,
    HuyetApTamTruong REAL,
    Cholesterol TEXT,
    DuongHuyet TEXT,
    HutThuoc INTEGER,
    UongCon INTEGER,
    TapTheDuc INTEGER,
    NguyCo TEXT,
    LoiKhuyen TEXT,
    Tuoi INTEGER,
    GioiTinh TEXT
);
CREATE TABLE TinNhanAI (
    ID INTEGER PRIMARY KEY,
    BenhNhanID INTEGER,
    NoiDung TEXT,
    PhanHoi TEXT,
    ThoiGian TEXT
);
""")

# 5) Nạp dữ liệu
nguoidung.to_sql("NguoiDung", sq, if_exists="append", index=False)
chandoan.to_sql("ChanDoan", sq, if_exists="append", index=False)
if not tinnhanai.empty:
    tinnhanai.to_sql("TinNhanAI", sq, if_exists="append", index=False)

# 6) Tạo các VIEW tương đương SQL Server
sq.executescript("""
CREATE VIEW V_BenhNhan AS
SELECT 
    ID AS BenhNhanID, HoTen, Email, GioiTinh, DienThoai, NgaySinh, DiaChi, NgayTao
FROM NguoiDung
WHERE Role = 'patient';

CREATE VIEW V_LichSuChanDoan AS
SELECT
    cd.ID AS ChanDoanID,
    cd.BenhNhanID,
    cd.BacSiID,
    bn.HoTen AS TenBenhNhan,
    COALESCE(cd.GioiTinh, bn.GioiTinh) AS GioiTinh,
    COALESCE(cd.Tuoi, CAST(strftime('%Y','now') - strftime('%Y', bn.NgaySinh) AS INT)) AS Tuoi,
    bs.HoTen AS TenBacSi,
    cd.NgayChanDoan,
    cd.BMI, cd.HuyetApTamThu, cd.HuyetApTamTruong, cd.Cholesterol, cd.DuongHuyet,
    cd.HutThuoc, cd.UongCon, cd.TapTheDuc, cd.NguyCo, cd.LoiKhuyen
FROM ChanDoan cd
JOIN NguoiDung bn ON cd.BenhNhanID = bn.ID
LEFT JOIN NguoiDung bs ON cd.BacSiID = bs.ID;

CREATE VIEW V_ThongKeBacSi AS
SELECT 
    bs.ID AS BacSiID,
    bs.HoTen AS TenBacSi,
    COUNT(cd.ID) AS SoCaChanDoan
FROM NguoiDung bs
LEFT JOIN ChanDoan cd ON cd.BacSiID = bs.ID
WHERE bs.Role = 'doctor'
GROUP BY bs.ID, bs.HoTen;
""")

sq.commit()
sq.close()
print("✅ Đã tạo cvd_app.db + view từ SQL Server.")
