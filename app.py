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
# Cấu hình app
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
# Cấu hình Gemini
# ==========================================
MODEL_NAME = "models/gemini-2.5-flash-lite"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@lru_cache(maxsize=128)
def get_ai_advice_cached(prompt: str) -> str:
    """Gọi Gemini để sinh lời khuyên"""
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
# Đăng nhập
# ==========================================
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

# ==========================================
# Trang chủ
# ==========================================
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

# ==========================================
# Trang chẩn đoán
# ==========================================
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
    ai_advice = None
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

            # ===== Gọi AI Gemini để lấy lời khuyên =====
            prompt = f"""
            Bạn là chuyên gia tim mạch.
            Dữ liệu bệnh nhân:
            - Tuổi: {age}
            - Giới tính: {gender}
            - BMI: {bmi}
            - Huyết áp: {systolic}/{diastolic}
            - Cholesterol: {chol}
            - Đường huyết: {glucose}
            - Hút thuốc: {'Có' if smoking else 'Không'}
            - Uống rượu: {'Có' if alcohol else 'Không'}
            - Tập thể dục: {'Có' if exercise else 'Không'}

            Hãy:
            1. Nhận xét nguy cơ tim mạch.
            2. Đưa ra lời khuyên về lối sống và chế độ dinh dưỡng phù hợp.
            """
            ai_advice = get_ai_advice_cached(prompt)

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
    return render_template(
        'diagnose.html',
        result=result,
        ai_advice=ai_advice,
        file_result=file_result,
        benhnhans=benhnhans
    )

# ==========================================
# Lịch sử chẩn đoán
# ==========================================
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

@app.route('/delete_history/<int:id>', methods=['POST'])
def delete_history(id):
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    cur = conn.cursor()

    try:
        if session.get('role') == 'doctor':
            # Bác sĩ xóa bất kỳ chẩn đoán nào
            cur.execute("DELETE FROM BenhAn WHERE ChanDoanID=?", (id,))
            cur.execute("DELETE FROM ChanDoan WHERE ID=?", (id,))
        else:
            # Bệnh nhân chỉ được xóa của mình
            cur.execute("DELETE FROM BenhAn WHERE ChanDoanID=? AND ChanDoanID IN (SELECT ID FROM ChanDoan WHERE BenhNhanID=?)", (id, session['user_id']))
            cur.execute("DELETE FROM ChanDoan WHERE ID=? AND BenhNhanID=?", (id, session['user_id']))

        conn.commit()
    except Exception as e:
        conn.rollback()
    finally:
        conn.close()

    return redirect(url_for('history'))

# ==========================================
# Bệnh án
# ==========================================
@app.route('/records')
def records():
    # Bắt buộc đăng nhập
    if 'user' not in session:
        return redirect(url_for('login'))

    # Chỉ cho bác sĩ truy cập
    if session.get('role') != 'doctor':
        flash("Chỉ bác sĩ mới được xem bệnh án.", "warning")
        return redirect(url_for('history'))

    # Lấy dữ liệu bệnh án
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM V_BenhAnChiTiet ORDER BY NgayCapNhat DESC")
    records = cur.fetchall()
    conn.close()

    return render_template('records.html', records=records)

# ==========================================
# Hồ sơ
# ==========================================
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
