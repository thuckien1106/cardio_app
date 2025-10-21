/* ============================================================
   💖 DATABASE ỨNG DỤNG CHẨN ĐOÁN TIM MẠCH - CVD_App
   Phiên bản: 2025.10
   Tác giả: Hồ Kiến Thức
   ============================================================ */
CREATE DATABASE CVD_App;
GO
USE CVD_App;
GO

/* ============================================================
   1️⃣ BẢNG NGƯỜI DÙNG
   ============================================================ */
CREATE TABLE NguoiDung (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    HoTen NVARCHAR(100) NOT NULL,
    Email VARCHAR(100) NOT NULL UNIQUE,
    MatKhau VARCHAR(100) NOT NULL,
    Role NVARCHAR(20) CHECK (Role IN ('admin','doctor','patient')) NOT NULL,
    NgaySinh DATE NULL,
    GioiTinh NVARCHAR(10) CHECK (GioiTinh IN (N'Nam', N'Nữ')) NULL,
    DienThoai VARCHAR(20) NULL,
    DiaChi NVARCHAR(200) NULL,
    NgayTao DATETIME DEFAULT GETDATE()
);
GO


/* ============================================================
   2️⃣ BẢNG CHẨN ĐOÁN
   ============================================================ */
CREATE TABLE ChanDoan (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    BenhNhanID INT NOT NULL,
    BacSiID INT NULL, -- Nếu bệnh nhân tự chẩn đoán => NULL
    NgayChanDoan DATETIME DEFAULT GETDATE(),
    BMI FLOAT NULL,
    HuyetApTamThu FLOAT NULL,
    HuyetApTamTruong FLOAT NULL,
    Cholesterol NVARCHAR(50) NULL,
    DuongHuyet NVARCHAR(50) NULL,
    HutThuoc BIT NULL,
    UongCon BIT NULL,
    TapTheDuc BIT NULL,
    NguyCo NVARCHAR(50) NULL, -- "Nguy cơ cao"/"Nguy cơ thấp"
    LoiKhuyen NVARCHAR(MAX) NULL,

    CONSTRAINT FK_ChanDoan_BenhNhan FOREIGN KEY (BenhNhanID)
        REFERENCES NguoiDung(ID) ON DELETE CASCADE,
    CONSTRAINT FK_ChanDoan_BacSi FOREIGN KEY (BacSiID)
        REFERENCES NguoiDung(ID)
);
GO
ALTER TABLE ChanDoan 
ADD Tuoi INT NULL,
    GioiTinh NVARCHAR(10) NULL;


/* ============================================================
   3️⃣ DỮ LIỆU MẪU BAN ĐẦU
   ============================================================ */

-- 👑 Admin hệ thống
INSERT INTO NguoiDung (HoTen, Email, MatKhau, Role, GioiTinh, NgaySinh, DienThoai, DiaChi)
VALUES
(N'Quản trị viên', 'admin@cvdapp.com', '123456', 'admin', N'Nam', '1970-01-01', '0909000000', N'Phòng quản trị hệ thống - TP.HCM');

-- 👩‍⚕️ Danh sách bác sĩ
INSERT INTO NguoiDung (HoTen, Email, MatKhau, Role, NgaySinh, GioiTinh, DienThoai, DiaChi)
VALUES
(N'Nguyễn Văn A', 'doctor1@cvdapp.com', '123456', 'doctor', '1980-05-10', N'Nam', '0909123456', N'140 Lê Trọng Tấn, Tân Phú, TP.HCM'),
(N'Trần Thị B', 'doctor2@cvdapp.com', '123456', 'doctor', '1985-11-22', N'Nữ', '0909555123', N'15 Nguyễn Huệ, Quận 1, TP.HCM');
GO

-- 🧍‍♂️ Danh sách bệnh nhân
INSERT INTO NguoiDung (HoTen, Email, MatKhau, Role, NgaySinh, GioiTinh, DienThoai, DiaChi)
VALUES
(N'Lê Minh C', 'patient1@cvdapp.com', '123456', 'patient', '1995-03-18', N'Nam', '0911222333', N'25 Lý Thường Kiệt, Quận Tân Bình, TP.HCM'),
(N'Phạm Thị D', 'patient2@cvdapp.com', '123456', 'patient', '1992-08-05', N'Nữ', '0933444555', N'78 Cách Mạng Tháng 8, Quận 3, TP.HCM'),
(N'Hoàng Văn E', 'patient3@cvdapp.com', '123456', 'patient', '2000-12-12', N'Nam', '0988777666', N'102 Lạc Long Quân, Quận 11, TP.HCM');
GO


/* ============================================================
   5️⃣ VIEW HIỂN THỊ DANH SÁCH BỆNH NHÂN
   ============================================================ */
CREATE OR ALTER VIEW V_BenhNhan AS
SELECT 
    ID AS BenhNhanID,
    HoTen, 
    Email,
    GioiTinh,
    DienThoai,
    NgaySinh,
    DiaChi,
    NgayTao
FROM NguoiDung
WHERE Role = 'patient';
GO


/* ============================================================
   6️⃣ VIEW HIỂN THỊ LỊCH SỬ CHẨN ĐOÁN CHI TIẾT
   ============================================================ */
CREATE OR ALTER VIEW V_LichSuChanDoan AS
SELECT
    cd.ID AS ChanDoanID,
    cd.BenhNhanID,
    cd.BacSiID,
    bn.HoTen AS TenBenhNhan,
    ISNULL(cd.GioiTinh, bn.GioiTinh) AS GioiTinh,   -- 👈 Ưu tiên dữ liệu nhập
    ISNULL(cd.Tuoi, DATEDIFF(YEAR, bn.NgaySinh, GETDATE())) AS Tuoi, -- 👈 Ưu tiên dữ liệu nhập
    bs.HoTen AS TenBacSi,
    cd.NgayChanDoan,
    cd.BMI,
    cd.HuyetApTamThu,
    cd.HuyetApTamTruong,
    cd.Cholesterol,
    cd.DuongHuyet,
    cd.HutThuoc,
    cd.UongCon,
    cd.TapTheDuc,
    cd.NguyCo,
    cd.LoiKhuyen
FROM ChanDoan cd
JOIN NguoiDung bn ON cd.BenhNhanID = bn.ID
LEFT JOIN NguoiDung bs ON cd.BacSiID = bs.ID;
GO



/* ============================================================
   7️⃣ VIEW THỐNG KÊ BÁC SĨ
   ============================================================ */
CREATE OR ALTER VIEW V_ThongKeBacSi AS
SELECT 
    bs.ID AS BacSiID,
    bs.HoTen AS TenBacSi,
    COUNT(cd.ID) AS SoCaChanDoan
FROM NguoiDung bs
LEFT JOIN ChanDoan cd ON cd.BacSiID = bs.ID
WHERE bs.Role = 'doctor'
GROUP BY bs.ID, bs.HoTen;
GO


/* ============================================================
   8️⃣ THỐNG KÊ TOÀN HỆ THỐNG (Dành cho Admin)
   ============================================================ */
-- Tổng số bác sĩ, bệnh nhân, lượt chẩn đoán
SELECT
    (SELECT COUNT(*) FROM NguoiDung WHERE Role='doctor') AS TongBacSi,
    (SELECT COUNT(*) FROM NguoiDung WHERE Role='patient') AS TongBenhNhan,
    (SELECT COUNT(*) FROM ChanDoan) AS TongChanDoan;
GO


/* ============================================================
   ✅ KIỂM TRA VIEW
   ============================================================ */
SELECT * FROM V_BenhNhan;
SELECT * FROM V_LichSuChanDoan;
SELECT * FROM V_ThongKeBacSi;
GO
  
