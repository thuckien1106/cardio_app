from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import os
from werkzeug.utils import secure_filename
import pyodbc
import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from functools import lru_cache

# ==========================================
# Cấu hình Flask
# ==========================================
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "cvdapp-secret-key")

# ==========================================
# Kết nối SQL Server
# ==========================================
def get_connection():
    return pyodbc.connect(
        "DRIVER={SQL Server};"
        "SERVER=HKT;"
        "DATABASE=CVD_App;"
        "UID=sa;"
        "PWD=123"
    )

# ==========================================
# Cấu hình Gemini AI
# ==========================================
MODEL_NAME = "models/gemini-2.5-flash-lite"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@lru_cache(maxsize=128)
def get_ai_advice_cached(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Không thể lấy lời khuyên AI: {e}"

# ==========================================
# Cấu hình upload
# ==========================================
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ==========================================
# Đăng ký
# ==========================================
@app.route('/register', methods=['GET', 'POST'])
def register():
    today = datetime.date.today().strftime('%Y-%m-%d')

    if request.method == 'POST':
        ho_ten = request.form.get('ho_ten')
        gioi_tinh = request.form.get('gioi_tinh')
        ngay_sinh = request.form.get('ngay_sinh')
        email = request.form.get('email')
        mat_khau = request.form.get('mat_khau')
        role = request.form.get('role')

        conn = get_connection()
        cur = conn.cursor()

        # Kiểm tra email đã tồn tại
        cur.execute("SELECT ID FROM NguoiDung WHERE Email=?", (email,))
        if cur.fetchone():
            conn.close()
            return render_template('register.html', error="Email đã được sử dụng!", today=today)

        try:
            cur.execute("""
                INSERT INTO NguoiDung (HoTen, GioiTinh, NgaySinh, Email, MatKhau, Role)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (ho_ten, gioi_tinh, ngay_sinh, email, mat_khau, role))
            conn.commit()
            conn.close()

            # Thông báo qua JavaScript alert
            return render_template(
                'register.html',
                success=True,      # báo hiệu đăng ký thành công
                today=today
            )

        except Exception as e:
            conn.rollback()
            conn.close()
            return render_template('register.html', error=f"Lỗi: {e}", today=today)

    return render_template('register.html', today=today)

# ==========================================
# Đăng nhập
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('username')
        pw = request.form.get('password')

        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT ID, HoTen, Role
            FROM NguoiDung
            WHERE Email=? AND MatKhau=?
        """, (email, pw))
        user = cur.fetchone()
        conn.close()

        if user:
            session['user_id'] = user[0]
            session['user'] = user[1]
            session['role'] = user[2]
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Sai tài khoản hoặc mật khẩu")

    return render_template('login.html')


# ==========================================
# Trang chủ
# ==========================================
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')


# ==========================================
# Chẩn đoán
# ==========================================
@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    cur = conn.cursor()

    # Danh sách bệnh nhân cho bác sĩ
    benhnhans = []
    if session.get('role') == 'doctor':
        cur.execute("SELECT ID, HoTen, GioiTinh, NgaySinh FROM NguoiDung WHERE Role='patient'")
        rows = cur.fetchall()
        for r in rows:
            if r.NgaySinh:
                ns_fmt = r.NgaySinh.strftime("%d/%m/%Y") if not isinstance(r.NgaySinh, str) else r.NgaySinh
            else:
                ns_fmt = "Chưa khai báo"
            benhnhans.append({
                "ID": r.ID,
                "HoTen": r.HoTen,
                "GioiTinh": r.GioiTinh,
                "NgaySinh": ns_fmt
            })

    result = None
    ai_advice = None
    file_result = None

    # -------- Nhập liệu từ form --------
    if request.method == 'POST' and 'predict_form' in request.form:
        try:
            benhnhan_id = int(request.form.get('benhnhan_id')) if session.get('role') == 'doctor' else session['user_id']

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

            # Tính BMI
            bmi = round(weight / ((height / 100) ** 2), 2)

            # Dự đoán nguy cơ (demo)
            risk = 0
            if systolic > 140 or diastolic > 90: risk += 1
            if chol == 'above_normal': risk += 1
            elif chol == 'high': risk += 2
            if glucose == 'above_normal': risk += 1
            elif glucose == 'high': risk += 2
            if bmi > 30: risk += 1
            if smoking: risk += 1
            if alcohol: risk += 1

            nguy_co = "Nguy cơ cao" if risk >= 3 else "Nguy cơ thấp"
            result = f"{nguy_co} (BMI: {bmi})"

            # Gọi AI
            prompt = f"""
            Bạn là bác sĩ tim mạch.
            Dữ liệu: Tuổi {age}, Giới tính {gender}, BMI {bmi},
            Huyết áp {systolic}/{diastolic}, Chol {chol}, Đường huyết {glucose},
            Hút thuốc {'Có' if smoking else 'Không'}, Rượu {'Có' if alcohol else 'Không'},
            Tập thể dục {'Có' if exercise else 'Không'}.
            Hãy đưa ra lời khuyên về lối sống và phòng tránh bệnh tim mạch.
            """
            ai_advice = get_ai_advice_cached(prompt)

            # Lưu vào DB
            bacsi_id = session['user_id'] if session.get('role') == 'doctor' else None
            cur.execute("""
                INSERT INTO ChanDoan
                (BenhNhanID, BacSiID, BMI, HuyetApTamThu, HuyetApTamTruong,
                 Cholesterol, DuongHuyet, HutThuoc, UongCon, TapTheDuc, NguyCo, NgayChanDoan)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
            """, (benhnhan_id, bacsi_id, bmi, systolic, diastolic, chol, glucose, smoking, alcohol, exercise, nguy_co))
            conn.commit()

        except Exception as e:
            flash(f"Lỗi nhập liệu: {e}", "danger")

    # -------- Upload file --------
    if request.method == 'POST' and 'data_file' in request.files:
        f = request.files['data_file']
        if f.filename != '':
            path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(path)
            df = pd.read_csv(path) if f.filename.endswith('.csv') else pd.read_excel(path)
            df['Prediction'] = ['Cao' if i % 2 == 0 else 'Thấp' for i in range(len(df))]
            file_result = df.to_html(classes='table table-bordered', index=False)

    conn.close()
    return render_template('diagnose.html', benhnhans=benhnhans, result=result, ai_advice=ai_advice, file_result=file_result)


# ==========================================
# Lịch sử chẩn đoán
# ==========================================
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    cur = conn.cursor()

    if session.get('role') == 'doctor':
        cur.execute("SELECT * FROM V_LichSuChanDoan ORDER BY NgayChanDoan DESC")
    else:
        cur.execute("""
            SELECT * FROM V_LichSuChanDoan
            WHERE ChanDoanID IN (
                SELECT ID FROM ChanDoan WHERE BenhNhanID = ?
            )
            ORDER BY NgayChanDoan DESC
        """, (session['user_id'],))

    records = cur.fetchall()
    conn.close()
    return render_template('history.html', records=records)

@app.route('/delete_history/<int:id>', methods=['POST'])
def delete_history(id):
    if 'user' not in session:
        return redirect(url_for('login'))

    # Chỉ cho bác sĩ được xóa
    if session.get('role') != 'doctor':
        return redirect(url_for('history'))

    conn = get_connection()
    cur = conn.cursor()

    try:
        # Xóa bệnh án liên quan trước
        cur.execute("DELETE FROM BenhAn WHERE ChanDoanID = ?", (id,))
        # Xóa chẩn đoán
        cur.execute("DELETE FROM ChanDoan WHERE ID = ?", (id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Lỗi xóa: {e}")
    finally:
        conn.close()

    return redirect(url_for('history'))


# ==========================================
# Bệnh án
# ==========================================
@app.route('/records')
def records():
    if 'user' not in session:
        return redirect(url_for('login'))

    if session.get('role') != 'doctor':
        flash("Chỉ bác sĩ mới được xem bệnh án.", "warning")
        return redirect(url_for('history'))

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM V_BenhAnChiTiet ORDER BY NgayCapNhat DESC")
    records = cur.fetchall()
    conn.close()
    return render_template('records.html', records=records)


# ==========================================
# Hồ sơ cá nhân
# ==========================================
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    cur = conn.cursor()

    if request.method == 'POST':
        cur.execute("""
            UPDATE NguoiDung
            SET HoTen=?, DienThoai=?, NgaySinh=?, GioiTinh=?, DiaChi=?
            WHERE ID=?
        """, (
            request.form.get('ho_ten'),
            request.form.get('dien_thoai'),
            request.form.get('ngay_sinh'),
            request.form.get('gioi_tinh'),
            request.form.get('dia_chi'),
            session['user_id']
        ))
        conn.commit()
        flash("Cập nhật hồ sơ thành công!", "success")

    cur.execute("""
        SELECT HoTen, Email, Role, DienThoai, NgaySinh, GioiTinh, DiaChi
        FROM NguoiDung WHERE ID=?
    """, (session['user_id'],))
    user_info = cur.fetchone()
    conn.close()
    return render_template('profile.html', user_info=user_info)


# ==========================================
# Đăng xuất
# ==========================================
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ==========================================
# Main
# ==========================================
if __name__ == '__main__':
    app.run(debug=True)
