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
from datetime import date
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    today = date.today().strftime('%Y-%m-%d')


    if request.method == 'POST':
        ho_ten = request.form.get('ho_ten')
        gioi_tinh = request.form.get('gioi_tinh')
        ngay_sinh = request.form.get('ngay_sinh')
        email = request.form.get('email')
        mat_khau = request.form.get('mat_khau')

        role = 'patient'   # Mặc định là bệnh nhân

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
# Chẩn đoán bệnh tim mạch + Giải thích SHAP
# ==========================================
@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    cur = conn.cursor()

    # Danh sách bệnh nhân (nếu là bác sĩ)
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

    # Biến khởi tạo
    result = None
    ai_advice = None
    file_result = None
    risk_percent = None
    risk_level = None
    shap_file = None
    threshold = float(request.form.get('threshold', 0.5))  # Mặc định 0.5

    chol_map = {'normal': 1, 'above_normal': 2, 'high': 3}
    gluc_map = {'normal': 1, 'above_normal': 2, 'high': 3}

    # ===== Xử lý nhập liệu tay =====
    if request.method == 'POST' and 'predict_form' in request.form:
        try:
            benhnhan_id = int(request.form.get('benhnhan_id')) if session.get('role') == 'doctor' else session['user_id']

            # Lấy dữ liệu form
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

            # ===== Dự đoán bằng mô hình =====
            if xgb_model:
                X = np.array([[age, gender, systolic, diastolic,
                               chol_map.get(chol, 1), gluc_map.get(glucose, 1),
                               smoking, alcohol, exercise, bmi]])
                prob = float(xgb_model.predict_proba(X)[0, 1])
                risk_percent = round(prob * 100, 1)
                risk_level = 'high' if prob >= threshold else 'low'
            else:
                # Heuristic fallback
                score = 0
                if systolic > 140 or diastolic > 90: score += 1
                if chol == 'above_normal': score += 1
                elif chol == 'high': score += 2
                if glucose == 'above_normal': score += 1
                elif glucose == 'high': score += 2
                if bmi > 30: score += 1
                if smoking: score += 1
                if alcohol: score += 1
                prob = score / 8
                risk_percent = round(prob * 100, 1)
                risk_level = 'high' if prob >= threshold else 'low'

            nguy_co_text = "Nguy cơ cao" if risk_level == 'high' else "Nguy cơ thấp"
            result = f"{nguy_co_text} - {risk_percent}%"

            # ===== Lời khuyên từ AI =====
            prompt = f"""
            Bạn là bác sĩ tim mạch.
            Dữ liệu: Tuổi {age}, Giới tính {gender_raw}, BMI {bmi},
            Huyết áp {systolic}/{diastolic}, Cholesterol {chol}, Đường huyết {glucose},
            Hút thuốc {'Có' if smoking else 'Không'}, Rượu {'Có' if alcohol else 'Không'},
            Tập thể dục {'Có' if exercise else 'Không'}.
            Ngưỡng dự đoán: {threshold}.
            Hãy đưa ra lời khuyên ngắn gọn, dễ hiểu cho bệnh nhân.
            """
            ai_advice = get_ai_advice_cached(prompt)

            # ===== Giải thích kết quả bằng SHAP =====
            if xgb_model:
                try:
                    explainer = shap.TreeExplainer(xgb_model)
                    shap_values = explainer.shap_values(X)
                    shap.summary_plot(
                        shap_values, X,
                        feature_names=['Tuổi', 'Giới tính', 'HATT', 'HATTr',
                                       'Cholesterol', 'Đường huyết', 'Hút thuốc',
                                       'Rượu bia', 'Tập thể dục', 'BMI'],
                        show=False
                    )

                    # Lưu hình vào static/images/
                    shap_dir = os.path.join(app.root_path, 'static', 'images')
                    os.makedirs(shap_dir, exist_ok=True)
                    file_name = f"shap_{benhnhan_id}.png"
                    shap_path = os.path.join(shap_dir, file_name)

                    plt.tight_layout()
                    plt.savefig(shap_path, bbox_inches='tight')
                    plt.close()

                    shap_file = file_name  # chỉ tên file để render
                except Exception as e:
                    print(f"⚠️ Lỗi khi tạo biểu đồ SHAP: {e}")

            # ===== Lưu kết quả vào DB =====
            bacsi_id = session['user_id'] if session.get('role') == 'doctor' else None
            cur.execute("""
                INSERT INTO ChanDoan
                (BenhNhanID, BacSiID, BMI, HuyetApTamThu, HuyetApTamTruong,
                 Cholesterol, DuongHuyet, HutThuoc, UongCon, TapTheDuc,
                 NguyCo, LoiKhuyen, NgayChanDoan)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
            """, (benhnhan_id, bacsi_id, bmi, systolic, diastolic,
                  chol, glucose, smoking, alcohol, exercise,
                  nguy_co_text, ai_advice))
            conn.commit()

        except Exception as e:
            flash(f"Lỗi nhập liệu: {e}", "danger")

    # ===== Upload file CSV/Excel =====
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
            required = ['age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
                        'smoke', 'alco', 'active', 'weight', 'height']
            missing = [c for c in required if c not in df.columns]

            if missing:
                file_result = f"<p class='text-danger'>Thiếu các cột: {', '.join(missing)}</p>"
            else:
                df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
                df['gender'] = df['gender'].map({'Nam': 1, 'Nữ': 0}).fillna(df['gender'])
                df['smoke'] = df['smoke'].map({'yes': 1, 'no': 0}).fillna(df['smoke'])
                df['alco'] = df['alco'].map({'yes': 1, 'no': 0}).fillna(df['alco'])
                df['active'] = df['active'].map({'yes': 1, 'no': 0}).fillna(df['active'])
                df['cholesterol'] = df['cholesterol'].map(chol_map).fillna(df['cholesterol'])
                df['gluc'] = df['gluc'].map(gluc_map).fillna(df['gluc'])

                if xgb_model:
                    X = df[['age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol',
                            'gluc', 'smoke', 'alco', 'active', 'bmi']]
                    proba = xgb_model.predict_proba(X)[:, 1]
                    df['Nguy_cơ_%'] = (proba * 100).round(1)
                    df['Kết_quả'] = ['Nguy cơ cao' if p >= threshold else 'Nguy cơ thấp' for p in proba]
                else:
                    df['Nguy_cơ_%'] = 0
                    df['Kết_quả'] = 'Chưa có mô hình'

                file_result = df[['age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol',
                                  'gluc', 'smoke', 'alco', 'active', 'bmi',
                                  'Nguy_cơ_%', 'Kết_quả']] \
                    .to_html(classes='table table-bordered table-striped', index=False)

    conn.close()
    return render_template('diagnose.html',
                           benhnhans=benhnhans,
                           result=result,
                           risk_percent=risk_percent,
                           risk_level=risk_level,
                           threshold=threshold,
                           ai_advice=ai_advice,
                           file_result=file_result,
                           shap_file=shap_file)

# ==========================================
# Lịch sử chẩn đoán
# ==========================================
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    cur = conn.cursor()

    try:
        if session.get('role') == 'doctor':
            # Bác sĩ xem tất cả
            cur.execute("""
                SELECT ChanDoanID, TenBenhNhan, GioiTinh, Tuoi,
                       NgayChanDoan, BMI, HuyetApTamThu, HuyetApTamTruong,
                       Cholesterol, DuongHuyet, HutThuoc, UongCon, TapTheDuc,
                       NguyCo, LoiKhuyen
                FROM V_LichSuChanDoan
                ORDER BY NgayChanDoan DESC
            """)
        else:
            # Bệnh nhân chỉ xem lịch sử của mình
            cur.execute("""
                SELECT ChanDoanID, TenBenhNhan, GioiTinh, Tuoi,
                       NgayChanDoan, BMI, HuyetApTamThu, HuyetApTamTruong,
                       Cholesterol, DuongHuyet, HutThuoc, UongCon, TapTheDuc,
                       NguyCo, LoiKhuyen
                FROM V_LichSuChanDoan
                WHERE ChanDoanID IN (
                    SELECT ID FROM ChanDoan WHERE BenhNhanID = ?
                )
                ORDER BY NgayChanDoan DESC
            """, (session['user_id'],))

        records = cur.fetchall()

    except Exception as e:
        flash(f"❌ Lỗi khi tải lịch sử: {e}", "danger")
        records = []

    finally:
        conn.close()

    return render_template('history.html', records=records)


# ==========================================
# Xóa chẩn đoán
# ==========================================
@app.route('/delete_history/<int:id>', methods=['POST'])
def delete_history(id):
    if 'user' not in session:
        return redirect(url_for('login'))

    if session.get('role') != 'doctor':
        flash("❌ Bạn không có quyền xóa.", "danger")
        return redirect(url_for('history'))

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("DELETE FROM ChanDoan WHERE ID = ?", (id,))
        conn.commit()
        flash("✅ Đã xóa bản ghi chẩn đoán.", "success")

    except Exception as e:
        conn.rollback()
        flash(f"❌ Lỗi khi xóa: {e}", "danger")

    finally:
        conn.close()

    return redirect(url_for('history'))


# ==========================================
# Chỉnh sửa lời khuyên (chỉ dành cho bác sĩ)
# ==========================================
@app.route('/edit_advice/<int:id>', methods=['POST'])
def edit_advice(id):
    if 'user' not in session or session.get('role') != 'doctor':
        flash("❌ Bạn không có quyền chỉnh sửa lời khuyên.", "danger")
        return redirect(url_for('login'))

    new_advice = request.form.get('loi_khuyen')

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            UPDATE ChanDoan
            SET LoiKhuyen = ?
            WHERE ID = ?
        """, (new_advice, id))
        conn.commit()
        

    except Exception as e:
        conn.rollback()
        

    finally:
        conn.close()

    return redirect(url_for('history'))

# ==========================================
# Quản lý tài khoản & hồ sơ bệnh nhân
# ==========================================
@app.route('/manage_accounts', methods=['GET', 'POST'])
def manage_accounts():
    # ✅ Chỉ cho phép bác sĩ truy cập
    if 'user' not in session or session.get('role') != 'doctor':
        return redirect(url_for('login'))

    conn = get_connection()
    cur = conn.cursor()

    # ================================
    # 🗑 XÓA tài khoản bệnh nhân
    # ================================
    if request.method == 'POST' and 'delete_patient' in request.form:
        patient_id = request.form.get('id')

        try:
            # Xóa toàn bộ lịch sử chẩn đoán trước
            cur.execute("DELETE FROM ChanDoan WHERE BenhNhanID=?", (patient_id,))
            # Xóa tài khoản bệnh nhân
            cur.execute("DELETE FROM NguoiDung WHERE ID=?", (patient_id,))
            conn.commit()
            flash("✅ Đã xóa tài khoản và toàn bộ lịch sử chẩn đoán của bệnh nhân.", "success")
        except Exception as e:
            conn.rollback()
            flash(f"❌ Lỗi khi xóa: {e}", "danger")

    # ================================
    # ✏️ CẬP NHẬT thông tin bệnh nhân
    # ================================
    if request.method == 'POST' and 'update_patient' in request.form:
        patient_id = request.form.get('id')

        try:
            cur.execute("""
                UPDATE NguoiDung
                SET HoTen = ?, GioiTinh = ?, NgaySinh = ?, DienThoai = ?, DiaChi = ?
                WHERE ID = ?
            """, (
                request.form.get('ho_ten'),
                request.form.get('gioi_tinh'),
                request.form.get('ngay_sinh'),
                request.form.get('dien_thoai'),
                request.form.get('dia_chi'),
                patient_id
            ))
            conn.commit()
            flash("✅ Đã cập nhật thông tin bệnh nhân.", "success")
        except Exception as e:
            conn.rollback()
            flash(f"❌ Lỗi khi cập nhật: {e}", "danger")

    # ================================
    # 🔎 TÌM KIẾM bệnh nhân
    # ================================
    search = request.args.get('search', '').strip()  # Lấy từ khóa tìm kiếm từ URL (?search=...)

    if search:
        cur.execute("""
            SELECT ID, HoTen, Email, GioiTinh, NgaySinh, DienThoai, DiaChi
            FROM NguoiDung
            WHERE Role = 'patient' AND (HoTen LIKE ? OR Email LIKE ?)
            ORDER BY HoTen
        """, (f"%{search}%", f"%{search}%"))
    else:
        cur.execute("""
            SELECT ID, HoTen, Email, GioiTinh, NgaySinh, DienThoai, DiaChi
            FROM NguoiDung
            WHERE Role = 'patient'
            ORDER BY HoTen
        """)

    raw_patients = cur.fetchall()

    # ================================
    # XỬ LÝ dữ liệu trả về
    # ================================
    patients = []
    for p in raw_patients:
        if p.NgaySinh and hasattr(p.NgaySinh, "strftime"):
            ngay_sinh_str = p.NgaySinh.strftime("%d/%m/%Y")
            ngay_sinh_val = p.NgaySinh.strftime("%Y-%m-%d")
        else:
            ngay_sinh_str = p.NgaySinh if p.NgaySinh else "—"
            ngay_sinh_val = p.NgaySinh if p.NgaySinh else ""

        patients.append({
            "ID": p.ID,
            "HoTen": p.HoTen,
            "Email": p.Email,
            "GioiTinh": p.GioiTinh,
            "NgaySinh_str": ngay_sinh_str,
            "NgaySinh_val": ngay_sinh_val,
            "DienThoai": p.DienThoai,
            "DiaChi": p.DiaChi
        })

    conn.close()

    # ✅ Truyền cả patients và từ khóa tìm kiếm vào template
    return render_template('manage_accounts.html', patients=patients, search=search)

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
# Trang thống kê
# ==========================================
@app.route('/stats')
def stats():
    if 'user' not in session or session.get('role') != 'doctor':
        return redirect(url_for('login'))

    conn = get_connection()
    cur = conn.cursor()

    # Lấy số lượng bác sĩ
    cur.execute("SELECT COUNT(*) FROM NguoiDung WHERE Role='doctor'")
    num_doctors = cur.fetchone()[0]

    # Lấy số lượng bệnh nhân
    cur.execute("SELECT COUNT(*) FROM NguoiDung WHERE Role='patient'")
    num_patients = cur.fetchone()[0]

    # Lấy số lượt chẩn đoán
    cur.execute("SELECT COUNT(*) FROM ChanDoan")
    num_diagnoses = cur.fetchone()[0]

    # Thống kê lượt chẩn đoán theo tháng (trong năm hiện tại)
    cur.execute("""
        SELECT FORMAT(NgayChanDoan, 'MM-yyyy') AS Thang,
               COUNT(*) AS SoLuot
        FROM ChanDoan
        GROUP BY FORMAT(NgayChanDoan, 'MM-yyyy')
        ORDER BY MIN(NgayChanDoan)
    """)
    rows = cur.fetchall()
    conn.close()

    # Tách dữ liệu thành 2 mảng cho biểu đồ
    month_labels = [r.Thang for r in rows]
    month_counts = [r.SoLuot for r in rows]

    return render_template(
        'stats.html',
        num_doctors=num_doctors,
        num_patients=num_patients,
        num_diagnoses=num_diagnoses,
        month_labels=month_labels,
        month_counts=month_counts
    )
# ==========================================
# Xuất file Excel thống kê hệ thống
# ==========================================
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from io import BytesIO
from flask import send_file
from datetime import datetime

@app.route('/export_stats')
def export_stats():
    if 'user' not in session or session.get('role') != 'doctor':
        return redirect(url_for('login'))

    conn = get_connection()
    cur = conn.cursor()

    # === Lấy thống kê tổng quan ===
    cur.execute("SELECT COUNT(*) FROM NguoiDung WHERE Role='doctor'")
    num_doctors = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM NguoiDung WHERE Role='patient'")
    num_patients = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM ChanDoan")
    num_diagnoses = cur.fetchone()[0]

    # === Thống kê theo tháng (SQL Server) ===
    cur.execute("""
        SELECT 
            RIGHT('0' + CAST(MONTH(NgayChanDoan) AS VARCHAR(2)), 2) AS Thang,
            COUNT(*) AS SoLuong
        FROM ChanDoan
        GROUP BY MONTH(NgayChanDoan)
        ORDER BY Thang
    """)
    month_data = cur.fetchall()
    conn.close()

    # === Tạo workbook Excel ===
    wb = Workbook()
    ws = wb.active
    ws.title = "Thống kê hệ thống"

    # 🎨 Định nghĩa style
    title_font = Font(size=16, bold=True, color="1F4E78")
    header_font = Font(size=12, bold=True, color="FFFFFF")
    normal_font = Font(size=11)
    center_align = Alignment(horizontal="center", vertical="center")
    border = Border(
        left=Side(style="thin", color="000000"),
        right=Side(style="thin", color="000000"),
        top=Side(style="thin", color="000000"),
        bottom=Side(style="thin", color="000000")
    )
    fill_blue = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    fill_gray = PatternFill(start_color="D9D9D9", end_color="D9D9D9", fill_type="solid")

    # === Tiêu đề chính ===
    ws.merge_cells("A1:D1")
    ws["A1"] = "BÁO CÁO THỐNG KÊ HỆ THỐNG CHẨN ĐOÁN TIM MẠCH"
    ws["A1"].font = title_font
    ws["A1"].alignment = center_align

    ws.append([])

    # === Thông tin chung ===
    ws["A3"] = "👨‍⚕️ Người lập báo cáo:"
    ws["B3"] = session.get('user')
    ws["A4"] = "📅 Ngày xuất báo cáo:"
    ws["B4"] = datetime.now().strftime("%d/%m/%Y %H:%M")
    ws.append([])

    # === Phần 1: Thống kê tổng quan ===
    ws.append(["HẠNG MỤC", "SỐ LƯỢNG"])
    for cell in ws[6]:
        cell.font = header_font
        cell.fill = fill_blue
        cell.alignment = center_align
        cell.border = border

    ws.append(["Tổng số bác sĩ", num_doctors])
    ws.append(["Tổng số bệnh nhân", num_patients])
    ws.append(["Tổng số lượt chẩn đoán", num_diagnoses])

    for row in ws.iter_rows(min_row=7, max_row=9, min_col=1, max_col=2):
        for cell in row:
            cell.font = normal_font
            cell.border = border
            if cell.col_idx == 2:
                cell.alignment = center_align

    ws.append([])
    ws.append([])

    # === Phần 2: Thống kê lượt chẩn đoán theo tháng ===
    ws.append(["THÁNG", "SỐ LƯỢT CHẨN ĐOÁN"])
    for cell in ws[12]:
        cell.font = header_font
        cell.fill = fill_blue
        cell.alignment = center_align
        cell.border = border

    start_row = 13
    for thang, soluong in month_data:
        ws.append([thang, soluong])

    for row in ws.iter_rows(min_row=start_row, max_row=start_row + len(month_data) - 1, min_col=1, max_col=2):
        for cell in row:
            cell.font = normal_font
            cell.border = border
            cell.alignment = center_align

    # === Căn chỉnh độ rộng cột ===
    ws.column_dimensions["A"].width = 35
    ws.column_dimensions["B"].width = 22
    ws.column_dimensions["C"].width = 15
    ws.column_dimensions["D"].width = 15

    # === Xuất file Excel ===
    output = BytesIO()
    wb.save(output)
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="BaoCao_ThongKe_HeThong.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

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
