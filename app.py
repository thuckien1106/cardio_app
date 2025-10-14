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
# 🔐 Đăng nhập hệ thống
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('username', '').strip().lower()
        pw = request.form.get('password', '').strip()

        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT ID, HoTen, Role
            FROM NguoiDung
            WHERE Email = ? AND MatKhau = ?
        """, (email, pw))
        user = cur.fetchone()
        conn.close()

        if user:
            session['user_id'] = user[0]
            session['user'] = user[1]
            session['role'] = user[2]

            # ✅ Điều hướng theo vai trò
            if user[2] == 'admin':
                return redirect(url_for('admin_manage_doctors'))
            else:
                return redirect(url_for('home'))
        else:
            flash("❌ Sai tài khoản hoặc mật khẩu. Vui lòng thử lại!", "danger")
            return render_template('login.html')

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

    # Danh sách bệnh nhân (chỉ dành cho bác sĩ)
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

    # ======== XỬ LÝ NHẬP LIỆU THỦ CÔNG ========
    if request.method == 'POST' and 'predict_form' in request.form:
        try:
            benhnhan_id = int(request.form.get('benhnhan_id')) if session.get('role') == 'doctor' else session['user_id']

            # Lấy dữ liệu nhập tay
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
                # Fallback khi không có mô hình
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

            # ===== Sinh lời khuyên AI =====
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

            # ===== Tạo biểu đồ SHAP =====
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

                    shap_dir = os.path.join(app.root_path, 'static', 'images')
                    os.makedirs(shap_dir, exist_ok=True)
                    shap_file = f"shap_{benhnhan_id}.png"
                    shap_path = os.path.join(shap_dir, shap_file)

                    plt.tight_layout()
                    plt.savefig(shap_path, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"⚠️ Lỗi khi tạo biểu đồ SHAP: {e}")

            # ===== Lưu kết quả vào CSDL =====
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

    # ======== UPLOAD FILE (CHỈ CHO BÁC SĨ) ========
    if session.get('role') == 'doctor' and request.method == 'POST' and 'data_file' in request.files:
        f = request.files['data_file']
        if f.filename != '':
            path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(path)

            # Đọc file
            df = pd.read_csv(path) if f.filename.endswith('.csv') else pd.read_excel(path)
            df.columns = [c.strip().lower() for c in df.columns]

            required = ['age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
                        'smoke', 'alco', 'active', 'weight', 'height']
            missing = [c for c in required if c not in df.columns]

            if missing:
                file_result = f"<p class='text-danger'>Thiếu các cột: {', '.join(missing)}</p>"
            else:
                # Tiền xử lý
                df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
                df['gender'] = df['gender'].map({'Nam': 1, 'Nữ': 0}).fillna(df['gender'])
                df['smoke'] = df['smoke'].map({'yes': 1, 'no': 0}).fillna(df['smoke'])
                df['alco'] = df['alco'].map({'yes': 1, 'no': 0}).fillna(df['alco'])
                df['active'] = df['active'].map({'yes': 1, 'no': 0}).fillna(df['active'])
                df['cholesterol'] = df['cholesterol'].map(chol_map).fillna(df['cholesterol'])
                df['gluc'] = df['gluc'].map(gluc_map).fillna(df['gluc'])

                # Dự đoán hàng loạt
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
                                  'Nguy_cơ_%', 'Kết_quả']].to_html(
                                      classes='table table-bordered table-striped', index=False)

    conn.close()
    return render_template(
        'diagnose.html',
        benhnhans=benhnhans,
        result=result,
        risk_percent=risk_percent,
        risk_level=risk_level,
        threshold=threshold,
        ai_advice=ai_advice,
        file_result=file_result,
        shap_file=shap_file
    )


# ==========================================
# 📜 Lịch sử chẩn đoán (có lọc & sắp xếp + đếm tổng số)
# ==========================================
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    cur = conn.cursor()

    # ===== Lấy các tham số lọc từ URL =====
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    doctor_id = request.args.get('doctor_id')
    risk_filter = request.args.get('risk_filter')
    sort_order = request.args.get('sort', 'desc')

    # ===== Điều kiện mặc định =====
    where_clause = "WHERE 1=1"
    params = []

    # Nếu là bệnh nhân → chỉ xem của họ
    if session.get('role') != 'doctor':
        where_clause += " AND BenhNhanID = ?"
        params.append(session['user_id'])

    # ===== Lọc theo ngày =====
    if start_date:
        where_clause += " AND NgayChanDoan >= CONVERT(DATE, ?)"
        params.append(start_date)
    if end_date:
        where_clause += " AND NgayChanDoan <= CONVERT(DATE, ?)"
        params.append(end_date)

    # ===== Lọc theo bác sĩ =====
    if doctor_id:
        where_clause += " AND BacSiID = ?"
        params.append(doctor_id)

    # ===== Lọc theo nguy cơ =====
    if risk_filter == 'high':
        where_clause += " AND LOWER(NguyCo) LIKE '%cao%'"
    elif risk_filter == 'low':
        where_clause += " AND LOWER(NguyCo COLLATE SQL_Latin1_General_Cp1253_CI_AI) LIKE '%thap%'"

    # ===== Câu truy vấn chính =====
    query = f"""
        SELECT ChanDoanID, TenBenhNhan, GioiTinh, Tuoi, NgayChanDoan,
               BMI, HuyetApTamThu, HuyetApTamTruong, Cholesterol,
               DuongHuyet, HutThuoc, UongCon, TapTheDuc, NguyCo,
               LoiKhuyen, TenBacSi
        FROM V_LichSuChanDoan
        {where_clause}
        ORDER BY NgayChanDoan {'DESC' if sort_order == 'desc' else 'ASC'}
    """

    cur.execute(query, params)
    records = cur.fetchall()
    conn.close()

    # ✅ Đếm tổng số bản ghi (phục vụ hiển thị trên giao diện)
    total_records = len(records)

    # ===== Lấy danh sách bác sĩ (nếu là bác sĩ đăng nhập) =====
    doctors = []
    if session.get('role') == 'doctor':
        conn2 = get_connection()
        cur2 = conn2.cursor()
        cur2.execute("SELECT ID, HoTen FROM NguoiDung WHERE Role='doctor'")
        doctors = cur2.fetchall()
        conn2.close()

    # ===== Render =====
    return render_template(
        'history.html',
        records=records,
        doctors=doctors,
        start_date=start_date,
        end_date=end_date,
        doctor_id=doctor_id,
        risk_filter=risk_filter,
        sort_order=sort_order,
        total_records=total_records   # 👈 Thêm dòng này
    )


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
    # ➕ THÊM bệnh nhân mới
    # ================================
    if request.method == 'POST' and 'add_patient' in request.form:
        ho_ten = request.form.get('ho_ten')
        gioi_tinh = request.form.get('gioi_tinh')
        ngay_sinh = request.form.get('ngay_sinh')
        email = request.form.get('email')
        mat_khau = request.form.get('mat_khau')
        dien_thoai = request.form.get('dien_thoai')
        dia_chi = request.form.get('dia_chi')

        try:
            cur.execute("""
                INSERT INTO NguoiDung (HoTen, GioiTinh, NgaySinh, Email, MatKhau, DienThoai, DiaChi, Role)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'patient')
            """, (ho_ten, gioi_tinh, ngay_sinh, email, mat_khau, dien_thoai, dia_chi))
            conn.commit()
            flash("✅ Đã thêm bệnh nhân mới thành công!", "success")
        except Exception as e:
            conn.rollback()
            flash(f"❌ Lỗi khi thêm bệnh nhân: {e}", "danger")

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
# 📤 Xuất báo cáo kết quả chẩn đoán ra Excel 
# ==========================================
@app.route('/export_diagnosis', methods=['POST'])
def export_diagnosis():
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as ExcelImage
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from io import BytesIO
    from flask import send_file
    from datetime import datetime
    import os

    # ===== Dữ liệu từ form =====
    data = {key: request.form.get(key, '') for key in [
        'age', 'gender', 'bmi', 'systolic', 'diastolic', 'cholesterol',
        'glucose', 'smoking', 'alcohol', 'exercise',
        'risk_percent', 'risk_level', 'ai_advice', 'shap_file', 'benhnhan_id'
    ]}

    # ===== Lấy tên người đăng nhập & vai trò =====
    user_name = session.get('user', 'Người dùng')
    user_role = session.get('role', 'patient')

    # ===== Xác định tên bệnh nhân và bác sĩ =====
    patient_name = None
    doctor_name = None

    if user_role == 'doctor':
        # Bác sĩ chọn bệnh nhân trong danh sách => lấy tên bệnh nhân từ DB
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT HoTen FROM NguoiDung WHERE ID = ?", data.get('benhnhan_id'))
        row = cur.fetchone()
        conn.close()
        patient_name = row[0] if row else "Không xác định"
        doctor_name = user_name
    else:
        # Bệnh nhân tự chẩn đoán
        patient_name = user_name
        doctor_name = "—"

    # ===== Tạo workbook =====
    wb = Workbook()
    ws = wb.active
    ws.title = "Báo cáo chẩn đoán"

    # ===== Style =====
    title_font = Font(size=18, bold=True, color="1F4E78")
    header_font = Font(size=13, bold=True, color="FFFFFF")
    section_font = Font(size=12, bold=True, color="1F4E78")
    normal_font = Font(size=11)
    advice_font = Font(size=12, color="000000")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left = Alignment(horizontal="left", vertical="center", wrap_text=True)
    wrap = Alignment(wrap_text=True, vertical="top")
    border = Border(
        left=Side(style="thin", color="000000"),
        right=Side(style="thin", color="000000"),
        top=Side(style="thin", color="000000"),
        bottom=Side(style="thin", color="000000")
    )
    fill_header = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    fill_sub = PatternFill(start_color="E9F3FF", end_color="E9F3FF", fill_type="solid")
    fill_high = PatternFill(start_color="F8D7DA", end_color="F8D7DA", fill_type="solid")
    fill_low = PatternFill(start_color="D1E7DD", end_color="D1E7DD", fill_type="solid")

    # ===== Tiêu đề =====
    ws.merge_cells("A1:E1")
    ws["A1"] = "BÁO CÁO KẾT QUẢ CHẨN ĐOÁN TIM MẠCH"
    ws["A1"].font = title_font
    ws["A1"].alignment = center
    ws.append([])

    # ===== I. Thông tin chung =====
    ws.merge_cells("A3:E3")
    ws["A3"] = "I. THÔNG TIN CHUNG"
    ws["A3"].font = section_font
    ws["A3"].alignment = left

    ws.append(["Tên bệnh nhân", patient_name])
    ws.append(["Bác sĩ chẩn đoán", doctor_name])
    ws.append(["Ngày tạo báo cáo", datetime.now().strftime("%d/%m/%Y %H:%M")])
    ws.append([])

    # ===== II. Dữ liệu đầu vào =====
    ws.merge_cells("A7:E7")
    ws["A7"] = "II. DỮ LIỆU ĐẦU VÀO"
    ws["A7"].font = section_font
    ws["A7"].alignment = left

    ws.append(["Thuộc tính", "Giá trị", "Thuộc tính", "Giá trị"])
    for cell in ws[8]:
        cell.font = header_font
        cell.fill = fill_header
        cell.border = border
        cell.alignment = center

    input_data = [
        ["Tuổi", data['age'], "Giới tính", data['gender']],
        ["BMI", data['bmi'], "Huyết áp (HATT/HATTr)", f"{data['systolic']}/{data['diastolic']}"],
        ["Cholesterol", data['cholesterol'], "Đường huyết", data['glucose']],
        ["Hút thuốc", "Có" if data['smoking']=="yes" else "Không", "Rượu/Bia", "Có" if data['alcohol']=="yes" else "Không"],
        ["Tập thể dục", "Có" if data['exercise']=="yes" else "Không", "", ""]
    ]
    for row in input_data:
        ws.append(row)
        for cell in ws[ws.max_row]:
            cell.font = normal_font
            cell.border = border
            cell.alignment = left

    ws.append([])

    # ===== III. Kết quả chẩn đoán =====
    ws.merge_cells(f"A{ws.max_row+1}:E{ws.max_row+1}")
    ws[f"A{ws.max_row}"] = "III. KẾT QUẢ CHẨN ĐOÁN"
    ws[f"A{ws.max_row}"].font = section_font
    ws[f"A{ws.max_row}"].alignment = left

    ws.append(["Nguy cơ", "Tỉ lệ (%)", "Đánh giá", ""])
    for cell in ws[ws.max_row]:
        cell.font = header_font
        cell.fill = fill_header
        cell.border = border
        cell.alignment = center

    ws.append([
        "Cao" if data['risk_level'] == 'high' else "Thấp",
        data['risk_percent'] + "%",
        "⚠️ Cần theo dõi" if data['risk_level'] == 'high' else "✅ Ổn định",
        ""
    ])
    for cell in ws[ws.max_row]:
        cell.font = normal_font
        cell.border = border
        cell.alignment = center
        cell.fill = fill_high if data['risk_level'] == 'high' else fill_low

    ws.append([])

    # ===== IV. Lời khuyên từ AI =====
    ws.merge_cells(f"A{ws.max_row+1}:E{ws.max_row+1}")
    ws[f"A{ws.max_row}"] = "IV. LỜI KHUYÊN TỪ AI"
    ws[f"A{ws.max_row}"].font = section_font
    ws[f"A{ws.max_row}"].alignment = left

    # ✅ Fix lỗi merged cell + format đẹp
    start_row = ws.max_row + 1
    end_row = start_row + 5
    ws.merge_cells(f"A{start_row}:E{end_row}")
    cell = ws[f"A{start_row}"]
    cell.value = data['ai_advice'] or "Chưa có lời khuyên từ AI."
    cell.alignment = wrap
    cell.font = advice_font
    cell.border = border
    cell.fill = fill_sub

    ws.append([])
    ws.append([])

    # ===== V. Biểu đồ SHAP =====
    shap_path = os.path.join(app.root_path, 'static', 'images', data['shap_file']) if data['shap_file'] else None
    if shap_path and os.path.exists(shap_path):
        ws.merge_cells(f"A{ws.max_row+1}:E{ws.max_row+1}")
        ws[f"A{ws.max_row}"] = "V. GIẢI THÍCH KẾT QUẢ BẰNG BIỂU ĐỒ SHAP"
        ws[f"A{ws.max_row}"].font = section_font
        ws[f"A{ws.max_row}"].alignment = left
        try:
            img = ExcelImage(shap_path)
            img.width = 520
            img.height = 320
            ws.add_image(img, f"A{ws.max_row+1}")
        except Exception as e:
            ws.append([f"Lỗi khi chèn hình: {e}"])

    # ===== Footer =====
    ws.append([])
    ws.merge_cells(f"A{ws.max_row}:E{ws.max_row}")
    ws[f"A{ws.max_row}"] = f"📅 Báo cáo được tạo bởi: {doctor_name or user_name} — {datetime.now().strftime('%H:%M, %d/%m/%Y')}"
    ws[f"A{ws.max_row}"].alignment = center
    ws[f"A{ws.max_row}"].font = Font(size=10, italic=True, color="777777")

    # ===== Căn chỉnh độ rộng =====
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 20
    ws.column_dimensions["C"].width = 25
    ws.column_dimensions["D"].width = 25
    ws.column_dimensions["E"].width = 10

    # ===== Xuất file =====
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    filename = f"BaoCao_ChanDoan_{patient_name.replace(' ', '_')}.xlsx"

    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ==========================================
# Đăng xuất
# ==========================================
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
# =========================================================
# 📊 DASHBOARD THỐNG KÊ (Admin)
# =========================================================
@app.route('/admin/dashboard')
def admin_dashboard():
    # --- Kiểm tra quyền truy cập ---
    if 'user' not in session or session.get('role') != 'admin':
        flash("Bạn không có quyền truy cập trang này!", "danger")
        return redirect(url_for('login'))

    conn = get_connection()
    cur = conn.cursor()

    # ==========================
    # 1️⃣ Tổng số bác sĩ, bệnh nhân, lượt chẩn đoán
    # ==========================
    cur.execute("SELECT COUNT(*) FROM NguoiDung WHERE Role='doctor'")
    total_doctors = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM NguoiDung WHERE Role='patient'")
    total_patients = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM ChanDoan")
    total_diagnoses = cur.fetchone()[0]

    # ==========================
    # 2️⃣ Lượt chẩn đoán theo tháng
    # ==========================
    cur.execute("""
        SELECT FORMAT(NgayChanDoan, 'MM-yyyy') AS Thang, COUNT(*) AS SoLuong
        FROM ChanDoan
        GROUP BY FORMAT(NgayChanDoan, 'MM-yyyy')
        ORDER BY MIN(NgayChanDoan)
    """)
    monthly = cur.fetchall()
    months = [row.Thang for row in monthly]
    counts = [row.SoLuong for row in monthly]

    # ==========================
    # 3️⃣ Tỷ lệ nguy cơ Cao / Thấp
    # ==========================
    cur.execute("""
        SELECT NguyCo, COUNT(*) AS SoLuong
        FROM ChanDoan
        GROUP BY NguyCo
    """)
    risk_data = cur.fetchall()
    risk_labels = [row.NguyCo for row in risk_data]
    risk_values = [row.SoLuong for row in risk_data]

    # ==========================
    # 4️⃣ Top 5 bác sĩ có nhiều ca nhất
    # ==========================
    cur.execute("""
        SELECT TOP 5 bs.HoTen, COUNT(cd.ID) AS SoCa
        FROM ChanDoan cd
        JOIN NguoiDung bs ON cd.BacSiID = bs.ID
        GROUP BY bs.HoTen
        ORDER BY SoCa DESC
    """)
    top_doctors = cur.fetchall()
    top_names = [row.HoTen for row in top_doctors]
    top_counts = [row.SoCa for row in top_doctors]

    # ==========================
    # 5️⃣ Trung bình chỉ số y khoa (BMI, Huyết áp, hành vi)
    # ==========================
    cur.execute("""
        SELECT 
            AVG(BMI) AS AvgBMI,
            AVG(HuyetApTamThu) AS AvgHATT,
            AVG(HuyetApTamTruong) AS AvgHATTr,
            SUM(CASE WHEN HutThuoc = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS SmokePercent,
            SUM(CASE WHEN UongCon = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS AlcoPercent,
            SUM(CASE WHEN TapTheDuc = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS ActivePercent
        FROM ChanDoan
    """)
    row = cur.fetchone()
    avg_bmi = round(row.AvgBMI, 1) if row.AvgBMI else 0
    avg_systolic = int(row.AvgHATT) if row.AvgHATT else 0
    avg_diastolic = int(row.AvgHATTr) if row.AvgHATTr else 0
    smoke_percent = round(row.SmokePercent, 1) if row.SmokePercent else 0
    alco_percent = round(row.AlcoPercent, 1) if row.AlcoPercent else 0
    active_percent = round(row.ActivePercent, 1) if row.ActivePercent else 0

    # ==========================
    # 6️⃣ Hiệu suất chẩn đoán của bác sĩ
    # ==========================
    cur.execute("""
        SELECT 
            ND.HoTen AS BacSi,
            COUNT(CD.ID) AS SoCa,
            AVG(CD.BMI) AS TB_BMI,
            SUM(CASE WHEN CD.NguyCo LIKE '%cao%' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS TyLeCao
        FROM ChanDoan CD
        JOIN NguoiDung ND ON CD.BacSiID = ND.ID
        GROUP BY ND.HoTen
        ORDER BY SoCa DESC
    """)
    perf_rows = cur.fetchall()
    perf_names = [row.BacSi for row in perf_rows]
    perf_cases = [row.SoCa for row in perf_rows]
    perf_rate = [round(row.TyLeCao, 1) if row.TyLeCao else 0 for row in perf_rows]

    conn.close()

    # ==========================
    # Trả dữ liệu cho template
    # ==========================
    return render_template(
        'admin_dashboard.html',
        total_doctors=total_doctors,
        total_patients=total_patients,
        total_diagnoses=total_diagnoses,
        months=months,
        counts=counts,
        risk_labels=risk_labels,
        risk_values=risk_values,
        top_names=top_names,
        top_counts=top_counts,
        avg_bmi=avg_bmi,
        avg_systolic=avg_systolic,
        avg_diastolic=avg_diastolic,
        smoke_percent=smoke_percent,
        alco_percent=alco_percent,
        active_percent=active_percent,
        perf_names=perf_names,
        perf_cases=perf_cases,
        perf_rate=perf_rate
    )


# =========================================================
# 👩‍⚕️ Quản lý Bác sĩ (Admin)
# =========================================================
@app.route('/admin/manage_doctors', methods=['GET', 'POST'])
def admin_manage_doctors():
    # -------------------- Kiểm tra quyền truy cập --------------------
    if 'user' not in session or session.get('role') != 'admin':
        flash("Bạn không có quyền truy cập trang này!", "danger")
        return redirect(url_for('login'))

    import datetime
    conn = get_connection()
    cur = conn.cursor()

    # ======================== 🟢 THÊM BÁC SĨ ========================
    if request.method == 'POST' and 'add_doctor' in request.form:
        ho_ten = request.form.get('ho_ten', '').strip()
        email = request.form.get('email', '').strip().lower()
        mat_khau = request.form.get('mat_khau', '').strip()
        gioi_tinh = request.form.get('gioi_tinh')
        ngay_sinh = request.form.get('ngay_sinh') or None
        dien_thoai = request.form.get('dien_thoai')
        dia_chi = request.form.get('dia_chi')

        # Kiểm tra trùng email
        cur.execute("SELECT COUNT(*) FROM NguoiDung WHERE Email = ?", (email,))
        if cur.fetchone()[0] > 0:
            flash("❌ Email này đã tồn tại trong hệ thống!", "danger")
        else:
            cur.execute("""
                INSERT INTO NguoiDung (HoTen, Email, MatKhau, Role, NgaySinh, GioiTinh, DienThoai, DiaChi)
                VALUES (?, ?, ?, 'doctor', ?, ?, ?, ?)
            """, (ho_ten, email, mat_khau, ngay_sinh, gioi_tinh, dien_thoai, dia_chi))
            conn.commit()
            flash("✅ Thêm bác sĩ mới thành công!", "success")

    # ======================== 🟡 SỬA BÁC SĨ ========================
    elif request.method == 'POST' and 'edit_doctor' in request.form:
        id = request.form.get('id')
        ho_ten = request.form.get('ho_ten', '').strip()
        gioi_tinh = request.form.get('gioi_tinh')
        ngay_sinh = request.form.get('ngay_sinh') or None
        email = request.form.get('email', '').strip().lower()
        mat_khau = request.form.get('mat_khau', '').strip()
        dien_thoai = request.form.get('dien_thoai')
        dia_chi = request.form.get('dia_chi')

        # Nếu không nhập mật khẩu → giữ nguyên mật khẩu cũ
        if not mat_khau:
            cur.execute("""
                UPDATE NguoiDung
                SET HoTen = ?, GioiTinh = ?, NgaySinh = ?, Email = ?, DienThoai = ?, DiaChi = ?
                WHERE ID = ? AND Role = 'doctor'
            """, (ho_ten, gioi_tinh, ngay_sinh, email, dien_thoai, dia_chi, id))
        else:
            cur.execute("""
                UPDATE NguoiDung
                SET HoTen = ?, GioiTinh = ?, NgaySinh = ?, Email = ?, MatKhau = ?, DienThoai = ?, DiaChi = ?
                WHERE ID = ? AND Role = 'doctor'
            """, (ho_ten, gioi_tinh, ngay_sinh, email, mat_khau, dien_thoai, dia_chi, id))

        conn.commit()
        flash("✏️ Cập nhật thông tin bác sĩ thành công!", "success")

    # ======================== 🔴 XÓA BÁC SĨ ========================
    elif request.method == 'POST' and 'delete_doctor' in request.form:
        id = request.form.get('id')
        cur.execute("DELETE FROM NguoiDung WHERE ID = ? AND Role = 'doctor'", (id,))
        conn.commit()
        flash("🗑 Đã xóa bác sĩ khỏi hệ thống!", "success")

    # ======================== 📋 HIỂN THỊ DANH SÁCH ========================
    cur.execute("""
        SELECT ID, HoTen, Email, GioiTinh, NgaySinh, DienThoai, DiaChi, NgayTao
        FROM NguoiDung
        WHERE Role = 'doctor'
        ORDER BY NgayTao DESC
    """)
    doctors = cur.fetchall()

    # ✅ Chuyển chuỗi ngày sang datetime (nếu SQL Server trả về dạng text)
    for d in doctors:
        # NgaySinh
        if hasattr(d, 'NgaySinh') and isinstance(d.NgaySinh, str):
            try:
                d.NgaySinh = datetime.datetime.strptime(d.NgaySinh.split(" ")[0], "%Y-%m-%d")
            except:
                d.NgaySinh = None

        # NgayTao
        if hasattr(d, 'NgayTao') and isinstance(d.NgayTao, str):
            try:
                d.NgayTao = datetime.datetime.strptime(d.NgayTao.split(" ")[0], "%Y-%m-%d")
            except:
                d.NgayTao = None

    conn.close()

    # Trả về giao diện
    return render_template('admin_doctors.html', doctors=doctors)

# ==========================================
# 📊 XUẤT FILE EXCEL THỐNG KÊ HỆ THỐNG (NÂNG CẤP)
# ==========================================
@app.route('/export_admin_stats', methods=['POST'])
def export_admin_stats():
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.chart import PieChart, BarChart, LineChart, Reference
    from openpyxl.utils import get_column_letter
    from io import BytesIO
    from flask import send_file
    from datetime import datetime

    conn = get_connection()
    cur = conn.cursor()

    # ===============================
    # Lấy dữ liệu thống kê từ DB
    # ===============================
    # Tổng quan
    cur.execute("SELECT COUNT(*) FROM NguoiDung WHERE Role='doctor'")
    total_doctors = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM NguoiDung WHERE Role='patient'")
    total_patients = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM ChanDoan")
    total_diagnoses = cur.fetchone()[0]

    # Tỷ lệ nguy cơ
    cur.execute("""
        SELECT NguyCo, COUNT(*) AS SoLuong
        FROM ChanDoan
        GROUP BY NguyCo
    """)
    risk_data = cur.fetchall()

    # Top 5 bác sĩ
    cur.execute("""
        SELECT TOP 5 bs.HoTen, COUNT(cd.ID) AS SoCa
        FROM ChanDoan cd
        JOIN NguoiDung bs ON cd.BacSiID = bs.ID
        GROUP BY bs.HoTen
        ORDER BY SoCa DESC
    """)
    top_doctors = cur.fetchall()

    # Hiệu suất bác sĩ
    cur.execute("""
        SELECT 
            ND.HoTen AS BacSi,
            COUNT(CD.ID) AS SoCa,
            SUM(CASE WHEN CD.NguyCo LIKE '%cao%' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS TyLeCao
        FROM ChanDoan CD
        JOIN NguoiDung ND ON CD.BacSiID = ND.ID
        GROUP BY ND.HoTen
        ORDER BY SoCa DESC
    """)
    perf_rows = cur.fetchall()

    conn.close()

    # ===============================
    # Tạo workbook Excel
    # ===============================
    wb = Workbook()
    ws = wb.active
    ws.title = "Tổng quan"

    # --- Style cơ bản ---
    title_font = Font(size=16, bold=True, color="1F4E78")
    header_font = Font(size=12, bold=True, color="FFFFFF")
    fill_blue = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    align_center = Alignment(horizontal="center", vertical="center")
    border = Border(
        left=Side(style="thin", color="999999"),
        right=Side(style="thin", color="999999"),
        top=Side(style="thin", color="999999"),
        bottom=Side(style="thin", color="999999")
    )

    # ===============================
    # 📄 Sheet 1: Tổng quan hệ thống
    # ===============================
    ws.merge_cells("A1:D1")
    ws["A1"] = "BÁO CÁO THỐNG KÊ HỆ THỐNG CHẨN ĐOÁN TIM MẠCH"
    ws["A1"].font = title_font
    ws["A1"].alignment = align_center

    ws.append([])
    ws.append(["Ngày xuất báo cáo:", datetime.now().strftime("%d/%m/%Y %H:%M")])
    ws.append([])
    ws.append(["Tổng số bác sĩ", total_doctors])
    ws.append(["Tổng số bệnh nhân", total_patients])
    ws.append(["Tổng số lượt chẩn đoán", total_diagnoses])
    ws.append([])

    ws.append(["Tên bác sĩ", "Số ca chẩn đoán"])
    for cell in ws[ws.max_row]:
        cell.font = header_font
        cell.fill = fill_blue
        cell.alignment = align_center
        cell.border = border
    for d in top_doctors:
        ws.append([d.HoTen, d.SoCa])
        for cell in ws[ws.max_row]:
            cell.border = border
    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 20

    # ===============================
    # 📊 Sheet 2: Bác sĩ / Bệnh nhân
    # ===============================
    ws2 = wb.create_sheet("Bác sĩ_Bệnh nhân")
    ws2.append(["Loại tài khoản", "Số lượng"])
    ws2.append(["Bác sĩ", total_doctors])
    ws2.append(["Bệnh nhân", total_patients])

    for cell in ws2[1]:
        cell.font = header_font
        cell.fill = fill_blue
        cell.alignment = align_center
        cell.border = border
    for row in ws2.iter_rows(min_row=2, max_col=2):
        for c in row:
            c.border = border
            c.alignment = align_center

    pie = PieChart()
    pie.title = "Tỷ lệ Bác sĩ / Bệnh nhân"
    data = Reference(ws2, min_col=2, min_row=1, max_row=3)
    labels = Reference(ws2, min_col=1, min_row=2, max_row=3)
    pie.add_data(data, titles_from_data=True)
    pie.set_categories(labels)
    ws2.add_chart(pie, "D5")

    # ===============================
    # 📊 Sheet 3: Tỷ lệ nguy cơ
    # ===============================
    ws3 = wb.create_sheet("Nguy cơ")
    ws3.append(["Mức nguy cơ", "Số lượng"])
    for r in risk_data:
        ws3.append([r.NguyCo, r.SoLuong])

    for cell in ws3[1]:
        cell.font = header_font
        cell.fill = fill_blue
        cell.alignment = align_center
        cell.border = border

    bar = BarChart()
    bar.title = "Tỷ lệ nguy cơ cao / thấp"
    data = Reference(ws3, min_col=2, min_row=1, max_row=ws3.max_row)
    cats = Reference(ws3, min_col=1, min_row=2, max_row=ws3.max_row)
    bar.add_data(data, titles_from_data=True)
    bar.set_categories(cats)
    bar.y_axis.title = "Số lượng"
    ws3.add_chart(bar, "D5")

    # ===============================
    # 📊 Sheet 4: Top 5 bác sĩ
    # ===============================
    ws4 = wb.create_sheet("Top 5 bác sĩ")
    ws4.append(["Tên bác sĩ", "Số ca chẩn đoán"])
    for d in top_doctors:
        ws4.append([d.HoTen, d.SoCa])

    for cell in ws4[1]:
        cell.font = header_font
        cell.fill = fill_blue
        cell.alignment = align_center
        cell.border = border

    chart4 = BarChart()
    chart4.title = "Top 5 bác sĩ chẩn đoán nhiều ca nhất"
    data = Reference(ws4, min_col=2, min_row=1, max_row=ws4.max_row)
    cats = Reference(ws4, min_col=1, min_row=2, max_row=ws4.max_row)
    chart4.add_data(data, titles_from_data=True)
    chart4.set_categories(cats)
    chart4.y_axis.title = "Số ca"
    ws4.add_chart(chart4, "D5")

    # ===============================
    # 📊 Sheet 5: Hiệu suất bác sĩ
    # ===============================
    ws5 = wb.create_sheet("Hiệu suất bác sĩ")
    ws5.append(["Bác sĩ", "Số ca", "Tỷ lệ nguy cơ cao (%)"])
    for p in perf_rows:
        ws5.append([p.BacSi, p.SoCa, round(p.TyLeCao or 0, 1)])

    for cell in ws5[1]:
        cell.font = header_font
        cell.fill = fill_blue
        cell.alignment = align_center
        cell.border = border

    linechart = LineChart()
    linechart.title = "Hiệu suất chẩn đoán và tỷ lệ nguy cơ cao"
    data_line = Reference(ws5, min_col=3, min_row=1, max_row=ws5.max_row)
    cats = Reference(ws5, min_col=1, min_row=2, max_row=ws5.max_row)
    linechart.add_data(data_line, titles_from_data=True)
    linechart.set_categories(cats)
    linechart.y_axis.title = "Tỷ lệ (%)"

    barchart = BarChart()
    data_bar = Reference(ws5, min_col=2, min_row=1, max_row=ws5.max_row)
    barchart.add_data(data_bar, titles_from_data=True)
    barchart.set_categories(cats)
    barchart.y_axis.title = "Số ca"

    # Gộp 2 biểu đồ (bar + line)
    linechart.y_axis.crosses = "max"
    barchart += linechart
    ws5.add_chart(barchart, "E5")

    # ===============================
    # Xuất file
    # ===============================
    output = BytesIO()
    wb.save(output)
    output.seek(0)

    filename = f"ThongKe_HeThong_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"

    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ==========================================
# 🌿 Trang Kiến thức Y học (cho bệnh nhân)
# ==========================================
@app.route('/tips')
def tips():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Chỉ cho phép bệnh nhân xem
    if session.get('role') != 'patient':
        flash("Chỉ bệnh nhân mới được truy cập trang này.", "warning")
        return redirect(url_for('home'))
    
    return render_template('tips.html')

# ==========================================
# Main
# ==========================================
if __name__ == '__main__':
    app.run(debug=True)
