from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import os
from werkzeug.utils import secure_filename
import pyodbc
import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from functools import lru_cache
import numpy as np

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
# Load mô hình XGBoost
# ==========================================
xgb_model = None
try:
    import xgboost as xgb
    MODEL_PATH = "xgb_T11_Final.json"
    if os.path.exists(MODEL_PATH):
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(MODEL_PATH)
        print("✅ Mô hình XGBoost đã load thành công.")
    else:
        print("⚠️ Không tìm thấy file mô hình, sẽ dùng heuristic.")
except Exception as e:
    print(f"⚠️ Không thể load mô hình XGBoost: {e}")
    xgb_model = None

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

        # Kiểm tra email trùng
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

            return render_template('register.html', success=True, today=today)
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

    # Lấy danh sách bệnh nhân cho bác sĩ
    benhnhans = []
    if session.get('role') == 'doctor':
        cur.execute("SELECT ID, HoTen, GioiTinh, NgaySinh FROM NguoiDung WHERE Role='patient'")
        for r in cur.fetchall():
            ns_fmt = r.NgaySinh.strftime("%d/%m/%Y") if hasattr(r.NgaySinh, 'strftime') else str(r.NgaySinh)
            benhnhans.append({
                "ID": r.ID,
                "HoTen": r.HoTen,
                "GioiTinh": r.GioiTinh,
                "NgaySinh": ns_fmt
            })

    result = None
    ai_advice = None
    file_result = None
    risk_percent = None
    risk_level = None

    chol_map = {'normal': 1, 'above_normal': 2, 'high': 3}
    gluc_map = {'normal': 1, 'above_normal': 2, 'high': 3}

    # ===== Nhập liệu tay =====
    if request.method == 'POST' and 'predict_form' in request.form:
        try:
            benhnhan_id = int(request.form.get('benhnhan_id')) if session.get('role') == 'doctor' else session['user_id']

            age = int(request.form.get('age'))
            gender_raw = request.form.get('gender')
            gender = 1 if gender_raw == 'Nam' else 0
            weight = float(request.form.get('weight'))
            height = float(request.form.get('height'))
            bmi = round(weight / ((height / 100) ** 2), 2)
            systolic = float(request.form.get('systolic'))
            diastolic = float(request.form.get('diastolic'))
            chol = request.form.get('cholesterol')
            glucose = request.form.get('glucose')

            smoking = 1 if request.form.get('smoking') == 'yes' else 0
            alcohol = 1 if request.form.get('alcohol') == 'yes' else 0
            exercise = 1 if request.form.get('exercise') == 'yes' else 0

            # Dự đoán bằng mô hình
            if xgb_model:
                X = np.array([[age, gender, systolic, diastolic,
                               chol_map.get(chol, 1), gluc_map.get(glucose, 1),
                               smoking, alcohol, exercise, bmi]])
                prob = float(xgb_model.predict_proba(X)[0, 1])
                risk_percent = round(prob * 100, 1)
                risk_level = 'high' if prob >= 0.5 else 'low'
            else:
                # fallback heuristic
                score = 0
                if systolic > 140 or diastolic > 90: score += 1
                if chol == 'above_normal': score += 1
                elif chol == 'high': score += 2
                if glucose == 'above_normal': score += 1
                elif glucose == 'high': score += 2
                if bmi > 30: score += 1
                if smoking: score += 1
                if alcohol: score += 1
                risk_percent = round(score / 8 * 100, 1)
                risk_level = 'high' if score >= 3 else 'low'

            nguy_co_text = "Nguy cơ cao" if risk_level == 'high' else "Nguy cơ thấp"
            result = f"{nguy_co_text} - {risk_percent}%"

            # Lời khuyên từ AI
            prompt = f"""
            Bạn là bác sĩ tim mạch.
            Dữ liệu: Tuổi {age}, Giới tính {gender_raw}, BMI {bmi},
            Huyết áp {systolic}/{diastolic}, Chol {chol}, Đường huyết {glucose},
            Hút thuốc {'Có' if smoking else 'Không'}, Rượu {'Có' if alcohol else 'Không'},
            Tập thể dục {'Có' if exercise else 'Không'}.
            Hãy đưa ra lời khuyên ngắn gọn cho bệnh nhân.
            """
            ai_advice = get_ai_advice_cached(prompt)

            # Lưu kết quả vào DB
            bacsi_id = session['user_id'] if session.get('role') == 'doctor' else None
            cur.execute("""
                INSERT INTO ChanDoan
                (BenhNhanID, BacSiID, BMI, HuyetApTamThu, HuyetApTamTruong,
                 Cholesterol, DuongHuyet, HutThuoc, UongCon, TapTheDuc, NguyCo, NgayChanDoan)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
            """, (benhnhan_id, bacsi_id, bmi, systolic, diastolic,
                  chol, glucose, smoking, alcohol, exercise, nguy_co_text))
            conn.commit()

        except Exception as e:
            flash(f"Lỗi nhập liệu: {e}", "danger")

    # ===== Upload file =====
    if request.method == 'POST' and 'data_file' in request.files:
        f = request.files['data_file']
        if f.filename != '':
            path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(path)

            if f.filename.endswith('.csv'):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)

            df.columns = [c.strip().lower() for c in df.columns]

            required = ['age','gender','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','weight','height']
            missing = [c for c in required if c not in df.columns]

            if missing:
                file_result = f"<p class='text-danger'>Thiếu các cột: {', '.join(missing)}</p>"
            else:
                # Tính BMI
                df['bmi'] = df['weight'] / ((df['height']/100) ** 2)

                # Chuyển đổi dữ liệu
                df['gender'] = df['gender'].map({'Nam':1, 'Nữ':0}).fillna(df['gender'])
                df['smoke'] = df['smoke'].map({'yes':1, 'no':0}).fillna(df['smoke'])
                df['alco'] = df['alco'].map({'yes':1, 'no':0}).fillna(df['alco'])
                df['active'] = df['active'].map({'yes':1, 'no':0}).fillna(df['active'])
                df['cholesterol'] = df['cholesterol'].map(chol_map).fillna(df['cholesterol'])
                df['gluc'] = df['gluc'].map(gluc_map).fillna(df['gluc'])

                # Dự đoán
                if xgb_model:
                    X = df[['age','gender','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','bmi']]
                    proba = xgb_model.predict_proba(X)[:,1] * 100
                    pred = (proba >= 50).astype(int)
                    df['Nguy_cơ_%'] = proba.round(1)
                    df['Kết_quả'] = ['Cao' if p==1 else 'Thấp' for p in pred]
                else:
                    df['Nguy_cơ_%'] = 0
                    df['Kết_quả'] = 'Chưa có mô hình'

                file_result = df[['age','gender','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','bmi','Nguy_cơ_%','Kết_quả']] \
                    .to_html(classes='table table-bordered table-striped', index=False)

    conn.close()
    return render_template('diagnose.html',
                           benhnhans=benhnhans,
                           result=result,
                           risk_percent=risk_percent,
                           risk_level=risk_level,
                           ai_advice=ai_advice,
                           file_result=file_result)

# ==========================================
# Lịch sử
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

    if session.get('role') != 'doctor':
        return redirect(url_for('history'))

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM BenhAn WHERE ChanDoanID=?", (id,))
        cur.execute("DELETE FROM ChanDoan WHERE ID=?", (id,))
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
        """, (request.form.get('ho_ten'),
              request.form.get('dien_thoai'),
              request.form.get('ngay_sinh'),
              request.form.get('gioi_tinh'),
              request.form.get('dia_chi'),
              session['user_id']))
        conn.commit()

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
