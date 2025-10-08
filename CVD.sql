-- Tạo Database
CREATE DATABASE CVD_App;
GO
USE CVD_App;
GO

/* ==========================
   Bảng Người Dùng
   ========================== */
CREATE TABLE NguoiDung (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    HoTen NVARCHAR(100) NOT NULL,
    Email VARCHAR(100) NOT NULL UNIQUE,
    MatKhau VARCHAR(100) NOT NULL,
    Role NVARCHAR(20) CHECK (Role IN ('doctor','patient')) NOT NULL,
    NgaySinh DATE NULL,
    GioiTinh NVARCHAR(10) CHECK (GioiTinh IN (N'Nam',N'Nữ')) NULL,
    DienThoai VARCHAR(20) NULL,
    DiaChi NVARCHAR(200) NULL,
    NgayTao DATETIME DEFAULT GETDATE()
);
GO

/* ==========================
   Bảng Lịch sử Chẩn đoán
   ========================== */
CREATE TABLE ChanDoan (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    BenhNhanID INT NOT NULL,
    BacSiID INT NULL,                      -- nếu bệnh nhân tự chẩn đoán thì NULL
    NgayChanDoan DATETIME DEFAULT GETDATE(),
    BMI FLOAT,
    HuyetApTamThu FLOAT,
    HuyetApTamTruong FLOAT,
    Cholesterol NVARCHAR(50),
    DuongHuyet NVARCHAR(50),
    HutThuoc BIT,
    UongCon BIT,
    TapTheDuc BIT,
    NguyCo NVARCHAR(50), -- Kết quả dự đoán (VD: Nguy cơ cao/thấp)
	LoiKhuyen NVARCHAR(MAX)
    FOREIGN KEY (BenhNhanID) REFERENCES NguoiDung(ID),
    FOREIGN KEY (BacSiID) REFERENCES NguoiDung(ID)
);
GO


-- Thêm bác sĩ
INSERT INTO NguoiDung (HoTen, Email, MatKhau, Role, NgaySinh, GioiTinh, DienThoai, DiaChi)
VALUES
(N'Nguyễn Văn A', 'doctor1@cvdapp.com', '123456', 'doctor', '1980-05-10', N'Nam', '0909123456', N'140 Lê Trọng Tấn, Tân Phú, TP.HCM'),
(N'Trần Thị B', 'doctor2@cvdapp.com', '123456', 'doctor', '1985-11-22', N'Nữ', '0909555123', N'15 Nguyễn Huệ, Quận 1, TP.HCM');
GO

-- Thêm bệnh nhân
INSERT INTO NguoiDung (HoTen, Email, MatKhau, Role, NgaySinh, GioiTinh, DienThoai, DiaChi)
VALUES
(N'Lê Minh C', 'patient1@cvdapp.com', '123456', 'patient', '1995-03-18', N'Nam', '0911222333', N'25 Lý Thường Kiệt, Quận Tân Bình, TP.HCM'),
(N'Phạm Thị D', 'patient2@cvdapp.com', '123456', 'patient', '1992-08-05', N'Nữ', '0933444555', N'78 Cách Mạng Tháng 8, Quận 3, TP.HCM'),
(N'Hoàng Văn E', 'patient3@cvdapp.com', '123456', 'patient', '2000-12-12', N'Nam', '0988777666', N'102 Lạc Long Quân, Quận 11, TP.HCM');
GO


/* View hiển thị danh sách tất cả bệnh nhân */
CREATE VIEW V_BenhNhan AS
SELECT ID AS BenhNhanID, HoTen, Email, GioiTinh, DienThoai, NgaySinh, NgayTao
FROM NguoiDung
WHERE Role = 'patient';
GO


/* View hiển thị lịch sử chẩn đoán chi tiết */
CREATE OR ALTER VIEW V_LichSuChanDoan AS
SELECT
    cd.ID AS ChanDoanID,
    bn.HoTen       AS TenBenhNhan,
    bn.GioiTinh,
    DATEDIFF(YEAR, bn.NgaySinh, GETDATE()) AS Tuoi,
    bs.HoTen       AS TenBacSi,
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
    cd.LoiKhuyen      -- Lời khuyên từ AI hoặc do bác sĩ nhập sau này
FROM ChanDoan cd
JOIN NguoiDung bn ON cd.BenhNhanID = bn.ID
LEFT JOIN NguoiDung bs ON cd.BacSiID = bs.ID;
GO


