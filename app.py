from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import os
from werkzeug.utils import secure_filename
import pyodbc
import datetime

# ==============================
# Khởi tạo Flask
# ==============================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "cvdapp-fixed-secret-key-2025")

# ==============================
# Kết nối SQL Server
# ==============================
def get_connection():
    conn = pyodbc.connect(
        "DRIVER={SQL Server};"
        "SERVER=HKT;"              # 👉 thay bằng tên server SQL của bạn
        "DATABASE=CVD_App;"
        "UID=sa;"
        "PWD=123"                  # 👉 thay bằng mật khẩu phù hợp
    )
    return conn

# ==============================
# Cấu hình upload
# ==============================
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==============================
# Đăng nhập
# ==============================
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('username')
        pw = request.form.get('password')

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ID, HoTen, Role
            FROM NguoiDung
            WHERE Email=? AND MatKhau=?
        """, (email, pw))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['user_id'] = user[0]
            session['user'] = user[1]
            session['role'] = user[2]
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Sai tài khoản hoặc mật khẩu")

    return render_template('login.html')

# ==============================
# Trang chủ
# ==============================
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

# ==============================
# Trang chẩn đoán
# ==============================
@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    cursor = conn.cursor()

    # Lấy danh sách bệnh nhân cho bác sĩ
    benhnhans = []
    if session.get('role') == 'doctor':
        cursor.execute("SELECT ID, HoTen, GioiTinh, NgaySinh FROM NguoiDung WHERE Role='patient'")
        rows = cursor.fetchall()

        # Chuyển đổi NgaySinh sang chuỗi dd/mm/yyyy hoặc “Chưa khai báo”
        for r in rows:
            if r.NgaySinh:
                if isinstance(r.NgaySinh, str):
                    try:
                        ns_date = datetime.datetime.strptime(r.NgaySinh, "%Y-%m-%d").date()
                        ns_fmt = ns_date.strftime("%d/%m/%Y")
                    except:
                        ns_fmt = r.NgaySinh
                else:
                    ns_fmt = r.NgaySinh.strftime("%d/%m/%Y")
            else:
                ns_fmt = "Chưa khai báo"

            benhnhans.append({
                "ID": r.ID,
                "HoTen": r.HoTen,
                "GioiTinh": r.GioiTinh,
                "NgaySinh": ns_fmt
            })

    result = None
    file_result = None

    # -------- Xử lý nhập liệu --------
    if request.method == 'POST' and 'predict_form' in request.form:
        try:
            # Nếu là bác sĩ → chọn bệnh nhân
            if session.get('role') == 'doctor':
                benhnhan_id = int(request.form.get('benhnhan_id'))
            else:
                benhnhan_id = session['user_id']

            age = int(request.form.get('age'))
            gender = request.form.get('gender')
            weight = float(request.form.get('weight'))
            height = float(request.form.get('height'))
            systolic = float(request.form.get('systolic'))
            diastolic = float(request.form.get('diastolic'))
            chol = request.form.get('cholesterol')
            glucose = request.form.get('glucose')

            smoking = 1 if request.form.get('smoking') == 'yes' else 0
            alcohol = 1 if request.form.get('alcohol') == 'yes' else 0
            exercise = 1 if request.form.get('exercise') == 'yes' else 0

            # ===== Tính BMI =====
            bmi = round(weight / ((height / 100) ** 2), 2)

            # ===== Logic dự đoán (demo) =====
            risk_score = 0
            if systolic > 140 or diastolic > 90:
                risk_score += 1
            if chol == 'above_normal':
                risk_score += 1
            elif chol == 'high':
                risk_score += 2
            if glucose == 'above_normal':
                risk_score += 1
            elif glucose == 'high':
                risk_score += 2
            if bmi > 30:
                risk_score += 1
            if smoking == 1:
                risk_score += 1
            if alcohol == 1:
                risk_score += 1

            nguy_co = "Nguy cơ cao" if risk_score >= 3 else "Nguy cơ thấp"
            result = f"{nguy_co} (BMI: {bmi})"

            # ===== Lưu vào DB =====
            bacsi_id = session['user_id'] if session.get('role') == 'doctor' else None
            cursor.execute("""
                INSERT INTO ChanDoan
                (BenhNhanID, BacSiID, BMI, HuyetApTamThu, HuyetApTamTruong,
                 Cholesterol, DuongHuyet, HutThuoc, UongCon, TapTheDuc, NguyCo, NgayChanDoan)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
            """, (
                benhnhan_id,
                bacsi_id,
                bmi, systolic, diastolic,
                chol, glucose,
                smoking, alcohol, exercise,
                nguy_co
            ))
            conn.commit()

        except Exception as e:
            flash(f"Lỗi nhập liệu: {e}", "danger")

    # -------- Xử lý upload file --------
    if request.method == 'POST' and 'data_file' in request.files:
        file = request.files['data_file']
        if file.filename != '':
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            if filename.endswith('.csv'):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)

            # Demo gán nhãn
            df['Prediction'] = ['Cao' if i % 2 == 0 else 'Thấp' for i in range(len(df))]
            file_result = df.to_html(classes='table table-striped table-bordered', index=False)

    conn.close()
    return render_template('diagnose.html', result=result, file_result=file_result, benhnhans=benhnhans)

# ==============================
# Trang lịch sử chẩn đoán
# ==============================
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    cursor = conn.cursor()

    if session.get('role') == 'doctor':
        cursor.execute("SELECT * FROM V_LichSuChanDoan ORDER BY NgayChanDoan DESC")
    else:
        cursor.execute("""
            SELECT * FROM V_LichSuChanDoan
            WHERE ChanDoanID IN (
                SELECT ID FROM ChanDoan WHERE BenhNhanID = ?
            )
            ORDER BY NgayChanDoan DESC
        """, (session['user_id'],))

    records = cursor.fetchall()
    conn.close()

    return render_template('history.html', records=records)

# ==============================
# Trang bệnh án
# ==============================
@app.route('/records')
def records():
    if 'user' not in session:
        return redirect(url_for('login'))

    if session.get('role') == 'patient':
        flash("Bạn không có quyền truy cập trang này.", "warning")
        return redirect(url_for('history'))

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM V_BenhAnChiTiet ORDER BY NgayCapNhat DESC")
    records = cursor.fetchall()
    conn.close()

    return render_template('records.html', records=records)

# ==============================
# Trang hồ sơ cá nhân
# ==============================
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    cursor = conn.cursor()

    if request.method == 'POST':
        ho_ten = request.form.get('ho_ten')
        dien_thoai = request.form.get('dien_thoai')
        ngay_sinh = request.form.get('ngay_sinh')
        gioi_tinh = request.form.get('gioi_tinh')
        dia_chi = request.form.get('dia_chi')

        cursor.execute("""
            UPDATE NguoiDung
            SET HoTen=?, DienThoai=?, NgaySinh=?, GioiTinh=?, DiaChi=?
            WHERE ID=?
        """, (ho_ten, dien_thoai, ngay_sinh, gioi_tinh, dia_chi, session['user_id']))
        conn.commit()
        flash("Cập nhật hồ sơ thành công!", "success")

    cursor.execute("""
        SELECT HoTen, Email, Role, DienThoai, NgaySinh, GioiTinh, DiaChi
        FROM NguoiDung WHERE ID=?
    """, (session['user_id'],))
    user_info = cursor.fetchone()
    conn.close()

    return render_template('profile.html', user_info=user_info)

# ==============================
# Đăng xuất
# ==============================
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ==============================
# Main
# ==============================
if __name__ == '__main__':
    app.run(debug=True)
