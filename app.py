from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify, abort
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
import base64
from email.message import EmailMessage
from authlib.integrations.flask_client import OAuth
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from secrets import token_urlsafe
import random
import re
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# Cáº¥u hÃ¬nh Flask
# ==========================================
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "cvdapp-secret-key")

oauth = OAuth(app)
SOCIAL_PROVIDERS = {"google": False}
DEFAULT_SOCIAL_PASSWORD = os.getenv("DEFAULT_SOCIAL_PASSWORD", "123456")
GMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@gmail\.com$", re.IGNORECASE)


def is_valid_gmail(email: str) -> bool:
    return bool(email and GMAIL_PATTERN.match(email.strip()))

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    SOCIAL_PROVIDERS["google"] = True

GMAIL_CLIENT_ID = os.getenv("GMAIL_CLIENT_ID_SEND")
GMAIL_CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET_SEND")
GMAIL_REFRESH_TOKEN = os.getenv("GMAIL_REFRESH_TOKEN")
GMAIL_SENDER = os.getenv("GMAIL_SENDER")
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def _build_gmail_service():
    if not (GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET and GMAIL_REFRESH_TOKEN and GMAIL_SENDER):
        raise RuntimeError("Chua cau hinh Gmail API (client id/secret, refresh token, sender).")
    creds = Credentials(
        None,
        refresh_token=GMAIL_REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GMAIL_CLIENT_ID,
        client_secret=GMAIL_CLIENT_SECRET,
        scopes=GMAIL_SCOPES,
    )
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def send_email(to_email: str, subject: str, html_body: str):
    """Gui email HTML thong qua Gmail API."""
    service = _build_gmail_service()

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = GMAIL_SENDER
    message["To"] = to_email
    message.set_content("Trinh duyet email cua ban khong ho tro noi dung HTML.")
    message.add_alternative(html_body, subtype="html")

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()

def get_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=PC1\\LNTUANDAT;"
        "DATABASE=CVD_App;"
        "Trusted_Connection=yes;"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )

@app.context_processor
def inject_social_flags():
    return {
        "social_google_enabled": SOCIAL_PROVIDERS.get("google", False),
    }

# ==========================================
# Cáº¥u hÃ¬nh Gemini AI
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
        return f" KhÃ´ng thá»ƒ láº¥y lá»i khuyÃªn AI: {e}"

# ==========================================
# Load mÃ´ hÃ¬nh XGBoost
# ==========================================
xgb_model = None
try:
    import xgboost as xgb
    MODEL_PATH = "xgb_p4.json"
    if os.path.exists(MODEL_PATH):
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(MODEL_PATH)
        print(" MÃ´ hÃ¬nh XGBoost Ä‘Ã£ load thÃ nh cÃ´ng.")
    else:
        print(" KhÃ´ng tÃ¬m tháº¥y file mÃ´ hÃ¬nh, sáº½ dÃ¹ng heuristic.")
except Exception as e:
    print(f" KhÃ´ng thá»ƒ load mÃ´ hÃ¬nh XGBoost: {e}")
    xgb_model = None
# Warm up mÃ´ hÃ¬nh ngay khi Flask khá»Ÿi Ä‘á»™ng
@app.before_request
def warmup_model():
    """Cháº¡y warm-up 1 láº§n duy nháº¥t khi nháº­n request Ä‘áº§u tiÃªn."""
    if not getattr(app, "_model_warmed", False):
        try:
            import numpy as np, shap
            dummy = np.array([[50,1,120,80,2,1,0,0,1,25]])
            _ = xgb_model.predict_proba(dummy)
            shap.TreeExplainer(xgb_model)
            print(" Warm-up hoÃ n táº¥t, model & SHAP Ä‘Ã£ cache.")
            app._model_warmed = True  # Ä‘Ã¡nh dáº¥u Ä‘Ã£ warm-up rá»“i
        except Exception as e:
            print(f" Warm-up model lá»—i: {e}")

# ==========================================
# Cáº¥u hÃ¬nh upload
# ==========================================
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import re

def _clear_pending_registration():
    session.pop('pending_registration', None)
    session.pop('pending_registration_code', None)
    session.pop('pending_registration_exp', None)

def _send_verification_email(name, email, code):
    html_body = render_template(
        'emails/verification_code.html',
        name=name or 'báº¡n',
        code=code
    )
    send_email(
        email,
        "MÃ£ xÃ¡c nháº­n Ä‘Äƒng kÃ½ CVD-App",
        html_body
    )

# ==========================================
# ðŸ§¾ ÄÄƒng kÃ½ tÃ i khoáº£n
# ==========================================
@app.route('/register', methods=['GET', 'POST'])
def register():
    today = date.today().strftime('%Y-%m-%d')

    if request.method == 'POST':
        ho_ten = request.form.get('ho_ten')
        gioi_tinh = request.form.get('gioi_tinh')
        ngay_sinh = request.form.get('ngay_sinh')
        email = request.form.get('email').strip().lower()
        mat_khau = request.form.get('mat_khau')
        mat_khau_confirm = request.form.get('mat_khau_confirm')

        if mat_khau != mat_khau_confirm:
            flash('Mat khau khong khop.', 'warning')
            return render_template('register.html', today=today)

        role = 'patient'

        if ngay_sinh:
            try:
                birth_date = datetime.datetime.strptime(ngay_sinh, "%Y-%m-%d").date()
                age = (date.today() - birth_date).days // 365
                if age < 16:
                    flash("Tuá»•i pháº£i tá»« 16 trá»Ÿ lÃªn.", "warning")
                    return render_template('register.html', today=today)
            except ValueError:
                flash("NgÃ y sinh khÃ´ng há»£p lá»‡.", "warning")
                return render_template('register.html', today=today)
        else:
            flash("Vui lÃ²ng nháº­p ngÃ y sinh.", "warning")
            return render_template('register.html', today=today)

        if not re.match(r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$', mat_khau):
            flash('Mat khau can it nhat 8 ky tu, gom chu hoa, so va ky tu dac biet.', 'warning')
            return render_template('register.html', today=today)

        conn = get_connection()
        cur = conn.cursor()
        cur.execute('SELECT ID FROM NguoiDung WHERE Email = ?', (email,))
        if cur.fetchone():
            conn.close()
            flash('Email da ton tai! Vui long chon email khac.', 'warning')
            return render_template('register.html', today=today)
        conn.close()

        verification_code = f"{random.randint(100000, 999999)}"
        session['pending_registration'] = {
            'ho_ten': ho_ten,
            'gioi_tinh': gioi_tinh,
            'ngay_sinh': ngay_sinh or None,
            'email': email,
            'mat_khau': mat_khau,
            'role': role
        }
        session['pending_registration_code'] = verification_code
        session['pending_registration_exp'] = (
            datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
        ).isoformat()

        try:
            _send_verification_email(ho_ten, email, verification_code)
            flash('Da gui ma xac nhan den email cua ban. Vui long kiem tra va nhap ma de hoan tat dang ky.', 'info')
            return redirect(url_for('verify_email'))
        except Exception as e:
            _clear_pending_registration()
            flash(f'Khong the gui email xac nhan: {e}', 'danger')
            return render_template('register.html', today=today)

    return render_template('register.html', today=today)

# ==========================================
# Xac thuc email dang ky
# ==========================================
@app.route('/verify-email', methods=['GET', 'POST'])
def verify_email():
    pending = session.get('pending_registration')
    if not pending:
        flash('Khong tim thay thong tin dang ky. Vui long dang ky lai.', 'warning')
        return redirect(url_for('register'))

    if request.method == 'POST':
        if request.form.get('action') == 'resend':
            new_code = f"{random.randint(100000, 999999)}"
            session['pending_registration_code'] = new_code
            session['pending_registration_exp'] = (
                datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
            ).isoformat()
            try:
                _send_verification_email(pending.get('ho_ten'), pending.get('email'), new_code)
                flash('Da gui lai ma xac nhan.', 'info')
            except Exception as e:
                flash(f'Khong the gui lai email: {e}', 'danger')
            return redirect(url_for('verify_email'))

        code = request.form.get('verification_code', '').strip()
        stored_code = session.get('pending_registration_code')
        expiry_str = session.get('pending_registration_exp')
        expiry = datetime.datetime.fromisoformat(expiry_str) if expiry_str else None

        if expiry and datetime.datetime.utcnow() > expiry:
            flash('Ma xac nhan da het han. Vui long yeu cau ma moi.', 'warning')
            return redirect(url_for('verify_email'))

        if not code or code != stored_code:
            flash('Ma xac nhan khong chinh xac.', 'danger')
            return redirect(url_for('verify_email'))

        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO NguoiDung (HoTen, GioiTinh, NgaySinh, Email, MatKhau, Role)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    pending['ho_ten'],
                    pending['gioi_tinh'],
                    pending['ngay_sinh'],
                    pending['email'],
                    pending['mat_khau'],
                    pending['role']
                )
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            conn.close()
            flash(f'Loi khi tao tai khoan: {e}', 'danger')
            return redirect(url_for('register'))
        else:
            conn.close()
            _clear_pending_registration()
            flash('Dang ky thanh cong! Vui long dang nhap.', 'success')
            return redirect(url_for('login'))

    email = pending.get('email')
    return render_template('verify_email.html', email=email)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('username', '').strip().lower()
        pw = request.form.get('password', '').strip()

        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT ID, HoTen, Role, MatKhau
            FROM NguoiDung
            WHERE Email = ?
        """, (email,))
        user = cur.fetchone()
        conn.close()

        # ðŸ”¹ Kiá»ƒm tra tÃ i khoáº£n & máº­t kháº©u
        if user and user.MatKhau == pw:
            # Táº¡o session
            session['user_id'] = user.ID
            session['user'] = user.HoTen
            session['role'] = user.Role

            # Hiá»ƒn thá»‹ thÃ´ng bÃ¡o chÃ o má»«ng
            flash(f"ðŸŽ‰ ChÃ o má»«ng {user.HoTen} Ä‘Äƒng nháº­p thÃ nh cÃ´ng!", "success")

            # âœ… Äiá»u hÆ°á»›ng theo vai trÃ²
            if user.Role == 'admin':
                return redirect(url_for('history'))
            elif user.Role == 'doctor':
                return redirect(url_for('home'))  
            else:
                return redirect(url_for('home'))

        else:
            # âŒ Sai máº­t kháº©u â†’ hiá»ƒn thá»‹ ngay
            flash("âŒ Sai tÃ i khoáº£n hoáº·c máº­t kháº©u. Vui lÃ²ng thá»­ láº¡i!", "danger")
            return render_template('login.html')

    # GET request â†’ hiá»ƒn thá»‹ form
    return render_template('login.html')

@app.route('/auth/<provider>')
def oauth_login(provider):
    provider = provider.lower()
    if provider not in SOCIAL_PROVIDERS:
        abort(404)
    if not SOCIAL_PROVIDERS.get(provider):
        flash("Chá»©c nÄƒng Ä‘Äƒng nháº­p nÃ y chÆ°a Ä‘Æ°á»£c cáº¥u hÃ¬nh.", "warning")
        return redirect(url_for('login'))

    client = oauth.create_client(provider)
    if not client:
        flash("KhÃ´ng tÃ¬m tháº¥y cáº¥u hÃ¬nh cho nhÃ  cung cáº¥p Ä‘Äƒng nháº­p.", "danger")
        return redirect(url_for('login'))

    redirect_uri = url_for('oauth_callback', provider=provider, _external=True)
    kwargs = {}
    if provider == "google":
        nonce = token_urlsafe(16)
        session['oauth_nonce'] = nonce
        kwargs['nonce'] = nonce
    return client.authorize_redirect(redirect_uri, **kwargs)

@app.route('/auth/callback/<provider>')
def oauth_callback(provider):
    provider = provider.lower()
    if provider not in SOCIAL_PROVIDERS or not SOCIAL_PROVIDERS.get(provider):
        flash("NhÃ  cung cáº¥p Ä‘Äƒng nháº­p chÆ°a Ä‘Æ°á»£c kÃ­ch hoáº¡t.", "warning")
        return redirect(url_for('login'))

    client = oauth.create_client(provider)
    if not client:
        flash("KhÃ´ng thá»ƒ khá»Ÿi táº¡o nhÃ  cung cáº¥p Ä‘Äƒng nháº­p.", "danger")
        return redirect(url_for('login'))

    try:
        token = client.authorize_access_token()
        nonce = session.pop('oauth_nonce', None)
        user_info = client.parse_id_token(token, nonce=nonce)
    except Exception as e:
        flash(f"KhÃ´ng thá»ƒ xÃ¡c thá»±c: {e}", "danger")
        return redirect(url_for('login'))

    email = user_info.get("email")
    if not email:
        flash("KhÃ´ng Ä‘á»c Ä‘Æ°á»£c email tá»« tÃ i khoáº£n cá»§a báº¡n. Vui lÃ²ng cho phÃ©p truy cáº­p email.", "warning")
        return redirect(url_for('login'))
    full_name = user_info.get("name") or user_info.get("given_name") or email.split("@")[0]

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT ID, HoTen, Role, MatKhau FROM NguoiDung WHERE Email = ?", (email,))
    user = cur.fetchone()

    if user:
        user_id = user.ID
        ho_ten = user.HoTen or full_name
        role = user.Role or 'patient'
        has_password = bool((user.MatKhau or "").strip())
    else:
        cur.execute("""
            INSERT INTO NguoiDung (HoTen, GioiTinh, NgaySinh, Email, MatKhau, Role)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (full_name, 'Nam', None, email, DEFAULT_SOCIAL_PASSWORD, 'patient'))
        conn.commit()
        cur.execute("SELECT ID, HoTen, Role, MatKhau FROM NguoiDung WHERE Email = ?", (email,))
        user = cur.fetchone()
        user_id = user.ID
        ho_ten = user.HoTen or full_name
        role = user.Role or 'patient'
        has_password = False

    needs_password_email = not has_password

    if needs_password_email:
        cur.execute("UPDATE NguoiDung SET MatKhau=? WHERE ID=?", (DEFAULT_SOCIAL_PASSWORD, user_id))
        conn.commit()
        conn.close()
        login_link = url_for('login', _external=True)

        email_body = f"""
        <div style="font-family:Arial,sans-serif;line-height:1.6">
          <h2 style="color:#0d6efd">ChÃ o {ho_ten},</h2>
          <p>Báº¡n vá»«a Ä‘Äƒng kÃ½/Ä‘Äƒng nháº­p báº±ng Google trÃªn CVD-App.</p>
          <p>Máº­t kháº©u Ä‘Äƒng nháº­p táº¡m thá»i cá»§a báº¡n lÃ :
            <strong style="font-size:1.2rem;">{DEFAULT_SOCIAL_PASSWORD}</strong></p>
          <p>Nháº¥n vÃ o liÃªn káº¿t sau Ä‘á»ƒ má»Ÿ trang Ä‘Äƒng nháº­p:
            <a href="{login_link}" style="color:#0d6efd; font-weight:bold;">Quay láº¡i Ä‘Äƒng nháº­p</a>
          </p>
          <p>Sau khi vÃ o há»‡ thá»‘ng, nhá»› Ä‘á»•i máº­t kháº©u trong pháº§n Há»“ sÆ¡ cÃ¡ nhÃ¢n Ä‘á»ƒ Ä‘áº£m báº£o an toÃ n.</p>
          <p>TrÃ¢n trá»ng,<br/>Äá»™i ngÅ© CVD-App</p>
        </div>
        """
        try:
            send_email(
                to_email=email,
                subject="Máº­t kháº©u Ä‘Äƒng nháº­p CVD-App",
                html_body=email_body
            )
        except Exception as e:
            print(f"[WARN] KhÃ´ng thá»ƒ gá»­i email máº­t kháº©u Google: {e}")

        return render_template('google_password_sent.html', email=email)

    conn.close()

    session['user_id'] = user_id
    session['user'] = ho_ten
    session['role'] = role
    flash(f"ðŸŽ‰ ChÃ o má»«ng {ho_ten} Ä‘Äƒng nháº­p thÃ nh cÃ´ng!", "success")

    if role == 'admin':
        return redirect(url_for('history'))
    return redirect(url_for('home'))


# ==========================================
# Trang chá»§
# ==========================================
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')
# ==========================================
# ðŸ“¡ API: Láº¥y thÃ´ng tin bá»‡nh nhÃ¢n tá»« há»“ sÆ¡ NguoiDung
# ==========================================
@app.route('/get_patient_info/<int:benhnhan_id>')
def get_patient_info(benhnhan_id):
    if 'user' not in session or session.get('role') != 'doctor':
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_connection()
    cur = conn.cursor()

    # ðŸ”¹ Láº¥y trá»±c tiáº¿p tuá»•i vÃ  giá»›i tÃ­nh tá»« há»“ sÆ¡ NguoiDung
    cur.execute("""
        SELECT 
            DATEDIFF(YEAR, NgaySinh, GETDATE()) AS Tuoi,
            GioiTinh
        FROM NguoiDung
        WHERE ID = ?
    """, (benhnhan_id,))

    row = cur.fetchone()
    conn.close()

    if row:
        return jsonify({
            "tuoi": row.Tuoi,
            "gioitinh": row.GioiTinh
        })
    else:
        return jsonify({"tuoi": None, "gioitinh": None})

# ==========================================
# ðŸ©º Cháº©n Ä‘oÃ¡n bá»‡nh tim máº¡ch + Giáº£i thÃ­ch SHAP
# ==========================================
@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    cur = conn.cursor()
    benhnhans = []
    if session.get('role') == 'doctor':
        cur.execute("SELECT ID, HoTen FROM NguoiDung WHERE Role='patient'")
        benhnhans = [
            {"ID": r.ID, "MaBN": f"BN{r.ID:03}", "HoTen": r.HoTen}
            for r in cur.fetchall()
        ]

    # --- Biáº¿n khá»Ÿi táº¡o ---
    result = None
    ai_advice = None
    file_result = None
    risk_percent = None
    risk_level = None
    shap_file = None
    results = []      
    threshold = float(request.form.get('threshold', 0.5))

    # ======================
    # ðŸ”¹ Xá»¬ LÃ NHáº¬P LIá»†U THá»¦ CÃ”NG
    # ======================
    if request.method == 'POST' and 'predict_form' in request.form:
        try:
            benhnhan_id = (
                int(request.form.get('benhnhan_id'))
                if session.get('role') == 'doctor'
                else session['user_id']
            )

            # --- Láº¥y dá»¯ liá»‡u nháº­p tay ---
            age = int(request.form.get('age'))
            gender_raw = request.form.get('gender')
            gender = 1 if gender_raw == 'Nam' else 0
            weight = float(request.form.get('weight'))
            height = float(request.form.get('height'))
            bmi = round(weight / ((height / 100) ** 2), 2)
            systolic = float(request.form.get('systolic'))
            diastolic = float(request.form.get('diastolic'))
            chol = int(request.form.get('cholesterol'))
            glucose = int(request.form.get('glucose'))
            smoking = 1 if request.form.get('smoking') == 'yes' else 0
            alcohol = 1 if request.form.get('alcohol') == 'yes' else 0
            exercise = 1 if request.form.get('exercise') == 'yes' else 0

            # --- Dá»± Ä‘oÃ¡n báº±ng mÃ´ hÃ¬nh ---
            if xgb_model:
                X = np.array([[age, gender, systolic, diastolic,
                               chol, glucose, smoking, alcohol, exercise, bmi]],
                             dtype=float)
                prob = float(xgb_model.predict_proba(X)[0, 1])
                risk_percent = round(prob * 100, 1)
                risk_level = 'high' if prob >= threshold else 'low'
            else:
                score = 0
                if systolic > 140 or diastolic > 90: score += 1
                score += chol
                score += glucose
                if bmi > 30: score += 1
                if smoking: score += 1
                if alcohol: score += 1
                prob = score / 8
                risk_percent = round(prob * 100, 1)
                risk_level = 'high' if prob >= threshold else 'low'

            nguy_co_text = "Nguy cÆ¡ cao" if risk_level == 'high' else "Nguy cÆ¡ tháº¥p"
            result = f"{nguy_co_text} - {risk_percent}%"

            # --- Sinh lá»i khuyÃªn AI ---
            chol_label = {0: "BÃ¬nh thÆ°á»ng", 1: "Cao nháº¹", 2: "Cao"}
            gluc_label = {0: "BÃ¬nh thÆ°á»ng", 1: "Cao nháº¹", 2: "Cao"}

            prompt = f"""
            Báº¡n lÃ  bÃ¡c sÄ© tim máº¡ch.
            Dá»¯ liá»‡u bá»‡nh nhÃ¢n:
            - Tuá»•i: {age}
            - Giá»›i tÃ­nh: {gender_raw}
            - BMI: {bmi}
            - Huyáº¿t Ã¡p: {systolic}/{diastolic}
            - Cholesterol: {chol_label.get(chol, 'KhÃ´ng rÃµ')}
            - ÄÆ°á»ng huyáº¿t: {gluc_label.get(glucose, 'KhÃ´ng rÃµ')}
            - HÃºt thuá»‘c: {'CÃ³' if smoking else 'KhÃ´ng'}
            - Uá»‘ng rÆ°á»£u bia: {'CÃ³' if alcohol else 'KhÃ´ng'}
            - Táº­p thá»ƒ dá»¥c: {'CÃ³' if exercise else 'KhÃ´ng'}

            NgÆ°á»¡ng dá»± Ä‘oÃ¡n: {threshold}.
            HÃ£y Ä‘Æ°a ra lá»i khuyÃªn ngáº¯n gá»n, dá»… hiá»ƒu, phÃ¹ há»£p vá»›i tÃ¬nh tráº¡ng trÃªn.
            """

            ai_advice_raw = get_ai_advice_cached(prompt)
            ai_advice = highlight_advice(ai_advice_raw)

            # --- Sinh biá»ƒu Ä‘á»“ SHAP ---
            if xgb_model:
                try:
                    explainer = shap.TreeExplainer(xgb_model)
                    shap_values = explainer.shap_values(X)
                    shap.summary_plot(
                        shap_values, X,
                        feature_names=[
                            'Tuá»•i', 'Giá»›i tÃ­nh', 'HATT', 'HATTr', 'Cholesterol',
                            'ÄÆ°á»ng huyáº¿t', 'HÃºt thuá»‘c', 'RÆ°á»£u bia', 'Táº­p thá»ƒ dá»¥c', 'BMI'
                        ],
                        show=False
                    )
                    shap_dir = os.path.join(app.root_path, 'static', 'images')
                    os.makedirs(shap_dir, exist_ok=True)
                    shap_file = f"shap_{benhnhan_id}.png"
                    plt.tight_layout()
                    plt.savefig(os.path.join(shap_dir, shap_file), bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"âš ï¸ Lá»—i khi táº¡o biá»ƒu Ä‘á»“ SHAP: {e}")

            # --- LÆ°u káº¿t quáº£ vÃ o CSDL ---
            chol_label = {0: "BÃ¬nh thÆ°á»ng", 1: "Cao nháº¹", 2: "Cao"}
            gluc_label = {0: "BÃ¬nh thÆ°á»ng", 1: "Cao nháº¹", 2: "Cao"}

            bacsi_id = session['user_id'] if session.get('role') == 'doctor' else None
            cur.execute("""
                INSERT INTO ChanDoan
                (BenhNhanID, BacSiID, Tuoi, GioiTinh, BMI, HuyetApTamThu, HuyetApTamTruong,
                Cholesterol, DuongHuyet, HutThuoc, UongCon, TapTheDuc,
                NguyCo, LoiKhuyen, NgayChanDoan)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
            """, (benhnhan_id, bacsi_id, age, gender_raw, bmi, systolic, diastolic,
                  chol_label.get(chol), gluc_label.get(glucose),
                  smoking, alcohol, exercise, nguy_co_text, ai_advice))
            conn.commit()

        except Exception as e:
            flash(f"Lá»—i nháº­p liá»‡u: {e}", "danger")

        # ======================
        # ðŸ”¹ Xá»¬ LÃ FILE CSV / EXCEL
        # ======================
    if request.method == 'POST' and 'data_file' in request.files:
        try:
            file = request.files['data_file']
            if not file:
                flash("âš ï¸ Vui lÃ²ng chá»n file CSV hoáº·c Excel trÆ°á»›c khi táº£i lÃªn.", "warning")
                return redirect(url_for('diagnose'))

            filename = file.filename.lower()
            if not filename.endswith(('.csv', '.xls', '.xlsx')):
                flash("âŒ Chá»‰ há»— trá»£ Ä‘á»‹nh dáº¡ng CSV, XLS hoáº·c XLSX", "danger")
                return redirect(url_for('diagnose'))

            # Äá»c file theo Ä‘á»‹nh dáº¡ng
            if filename.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            # Chuáº©n hÃ³a tÃªn cá»™t
            df.columns = [c.strip().lower() for c in df.columns]

            # CÃ¡c cá»™t báº¯t buá»™c
            required_cols = ['age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol',
                            'gluc', 'smoke', 'alco', 'active', 'weight', 'height']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                flash(f"âš ï¸ File thiáº¿u cÃ¡c cá»™t: {', '.join(missing)}", "danger")
                return redirect(url_for('diagnose'))

            # TÃ­nh BMI
            df['bmi'] = (df['weight'] / ((df['height'] / 100) ** 2)).round(2)

            results = []
            for _, row in df.iterrows():
                age = int(row['age'])
                gender_raw = row['gender']
                gender = 1 if str(gender_raw).strip().lower() in ['nam', 'male', '1'] else 0
                systolic = float(row['ap_hi'])
                diastolic = float(row['ap_lo'])
                chol = int(row['cholesterol'])
                gluc = int(row['gluc'])
                smoking = int(row['smoke'])
                alcohol = int(row['alco'])
                exercise = int(row['active'])
                bmi = float(row['bmi'])

                # Dá»± Ä‘oÃ¡n
                if xgb_model:
                    X = np.array([[age, gender, systolic, diastolic,
                                chol, gluc, smoking, alcohol, exercise, bmi]], dtype=float)
                    prob = float(xgb_model.predict_proba(X)[0, 1])
                else:
                    prob = 0.5

                risk_percent = round(prob * 100, 1)
                risk_level = "Nguy cÆ¡ cao" if prob >= threshold else "Nguy cÆ¡ tháº¥p"

                results.append({
                    "Tuá»•i": age,
                    "Giá»›i tÃ­nh": gender_raw,
                    "Huyáº¿t Ã¡p": f"{systolic}/{diastolic}",
                    "Cholesterol": chol,
                    "ÄÆ°á»ng huyáº¿t": gluc,
                    "BMI": bmi,
                    "HÃºt thuá»‘c": "CÃ³" if smoking else "KhÃ´ng",
                    "RÆ°á»£u/Bia": "CÃ³" if alcohol else "KhÃ´ng",
                    "Táº­p thá»ƒ dá»¥c": "CÃ³" if exercise else "KhÃ´ng",
                    "Nguy cÆ¡": risk_level,
                    "XÃ¡c suáº¥t (%)": risk_percent
                })

            file_result = pd.DataFrame(results).to_html(
                index=False,
                classes="table table-hover table-striped text-center align-middle small shadow-sm rounded-3"
            )

            flash("âœ… Dá»± Ä‘oÃ¡n tá»« file CSV/Excel Ä‘Ã£ hoÃ n táº¥t!", "success")

        except Exception as e:
            flash(f"âŒ Lá»—i khi xá»­ lÃ½ file CSV/Excel: {e}", "danger")


    conn.close()
    has_result = bool(result or ai_advice or shap_file or file_result)
    return render_template(
        'diagnose.html',
        benhnhans=benhnhans,
        result=result,
        risk_percent=risk_percent,
        risk_level=risk_level,
        threshold=threshold,
        ai_advice=ai_advice,
        file_result=file_result,
        shap_file=shap_file ,
        results=results,
        has_result=has_result 
    )

@app.route('/send-diagnosis-email', methods=['POST'])
def send_diagnosis_email():
    if 'user' not in session:
        return redirect(url_for('login'))

    risk_percent = request.form.get('risk_percent')
    risk_level = request.form.get('risk_level', 'low')
    threshold = request.form.get('threshold', '0.5')
    ai_advice_plain = request.form.get('ai_advice_plain', '').strip()
    benhnhan_id = request.form.get('benhnhan_id') or session.get('user_id')

    if not risk_percent:
        flash("ChÆ°a cÃ³ káº¿t quáº£ cháº©n Ä‘oÃ¡n Ä‘á»ƒ gá»­i email.", "warning")
        return redirect(url_for('diagnose'))

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT HoTen, Email FROM NguoiDung WHERE ID = ?", (benhnhan_id,))
    patient = cur.fetchone()
    conn.close()

    if not patient or not patient.Email:
        flash("KhÃ´ng tÃ¬m tháº¥y email ngÆ°á»i nháº­n.", "danger")
        return redirect(url_for('diagnose'))

    patient_name = patient.HoTen or "báº¡n"
    risk_text = "Nguy cÆ¡ cao" if risk_level == 'high' else "Nguy cÆ¡ tháº¥p"
    try:
        html_body = render_template(
            'emails/diagnosis_email.html',
            name=patient_name,
            risk_percent=risk_percent,
            risk_text=risk_text,
            threshold=threshold,
            ai_advice=ai_advice_plain,
            doctor_name=session.get('user', 'CVD-App')
        )
        send_email(
            patient.Email,
            "Káº¿t quáº£ cháº©n Ä‘oÃ¡n tim máº¡ch tá»« CVD-App",
            html_body
        )
        flash("ÄÃ£ gá»­i káº¿t quáº£ qua email. Cáº£m Æ¡n báº¡n Ä‘Ã£ sá»­ dá»¥ng dá»‹ch vá»¥!", "success")
    except Exception as e:
        flash(f"KhÃ´ng thá»ƒ gá»­i email: {e}", "danger")

    return redirect(url_for('diagnose'))

# ==========================================
# ðŸ§  HÃ m tÃ´ Ä‘áº­m lá»i khuyÃªn AI (1 mÃ u nháº¥n - FIX BUG "600;'>")
# ==========================================
import re

def highlight_advice(text):
    """ðŸ’¡ LÃ m ná»•i báº­t Ã½ chÃ­nh trong lá»i khuyÃªn AI chá»‰ vá»›i 1 mÃ u nháº¥n, an toÃ n khÃ´ng lá»—i HTML."""
    if not text:
        return ""

    # XÃ³a kÃ½ tá»± markdown (** hoáº·c *)
    text = re.sub(r'\*{1,3}', '', text)

    # ðŸ”¹ Nháº¥n máº¡nh tá»« khÃ³a (tÃ­ch cá»±c hoáº·c cáº£nh bÃ¡o)
    keywords = [
        r"(hÃ£y|nÃªn|cáº§n|duy trÃ¬|giá»¯|kiá»ƒm soÃ¡t|theo dÃµi|trÃ¡nh|khÃ´ng nÃªn|quan trá»ng|nguy cÆ¡|cao|bÃ©o phÃ¬|hÃºt thuá»‘c|rÆ°á»£u|bia|ngá»§ Ä‘á»§|táº­p luyá»‡n|Äƒn uá»‘ng|Ä‘iá»u chá»‰nh)"
    ]

    for kw in keywords:
        text = re.sub(
            kw,
            lambda m: f"<b class='text-primary fw-semibold'>{m.group(0)}</b>",
            text,
            flags=re.IGNORECASE
        )

    # ðŸ”¹ LÃ m ná»•i báº­t cÃ¡c con sá»‘ / pháº§n trÄƒm / Ä‘Æ¡n vá»‹ Ä‘o
    text = re.sub(
        r"\b\d+(\.\d+)?\s*(%|mmHg|kg|cm)?\b",
        lambda m: f"<b class='text-primary'>{m.group(0)}</b>",
        text
    )

    # ðŸ”¹ Thay newline báº±ng <br> cho trÃ¬nh bÃ y Ä‘áº¹p
    text = re.sub(r'\n+', '<br>', text.strip())

    # ðŸ”¹ GÃ³i khá»‘i ná»™i dung
    text = f"""
    <div style="
        text-align: justify;
        line-height: 1.8;
        font-size: 15px;
        color: #212529;
    ">
        {text}
    </div>
    """

    return text

# ==========================================
# ðŸ“œ Lá»‹ch sá»­ cháº©n Ä‘oÃ¡n (phÃ¢n quyá»n + lá»c bá»‡nh nhÃ¢n cho bÃ¡c sÄ©)
# ==========================================
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    cur = conn.cursor()

    # ===== Láº¥y cÃ¡c tham sá»‘ lá»c tá»« URL =====
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    patient_id = request.args.get('patient_id')
    doctor_id = request.args.get('doctor_id')
    risk_filter = request.args.get('risk_filter')
    sort_order = request.args.get('sort', 'desc')

    # ===== Äiá»u kiá»‡n máº·c Ä‘á»‹nh =====
    where_clause = "WHERE 1=1"
    params = []

    # ===== PhÃ¢n quyá»n =====
    role = session.get('role')

    if role == 'doctor':
        # ðŸ‘¨â€âš•ï¸ BÃ¡c sÄ© xem cÃ¡c ca do mÃ¬nh cháº©n Ä‘oÃ¡n
        where_clause += " AND BacSiID = ?"
        params.append(session['user_id'])
        # VÃ  cÃ³ thá»ƒ lá»c thÃªm theo bá»‡nh nhÃ¢n
        if patient_id:
            where_clause += " AND BenhNhanID = ?"
            params.append(patient_id)

    elif role == 'patient':
        # ðŸ§‘â€ðŸ¦± Bá»‡nh nhÃ¢n xem toÃ n bá»™ cÃ¡c ca cá»§a mÃ¬nh
        where_clause += " AND BenhNhanID = ?"
        params.append(session['user_id'])

    else:
        # ðŸ§‘â€ðŸ’¼ Admin xem toÃ n bá»™, cÃ³ thá»ƒ lá»c theo bÃ¡c sÄ© hoáº·c bá»‡nh nhÃ¢n
        if doctor_id:
            where_clause += " AND BacSiID = ?"
            params.append(doctor_id)
        if patient_id:
            where_clause += " AND BenhNhanID = ?"
            params.append(patient_id)

    # ===== Lá»c theo ngÃ y =====
    if start_date:
        where_clause += " AND NgayChanDoan >= CONVERT(DATE, ?)"
        params.append(start_date)
    if end_date:
        where_clause += " AND NgayChanDoan <= CONVERT(DATE, ?)"
        params.append(end_date)

    # ===== Lá»c theo nguy cÆ¡ =====
    if risk_filter == 'high':
        where_clause += " AND LOWER(NguyCo) LIKE '%cao%'"
    elif risk_filter == 'low':
        where_clause += " AND LOWER(NguyCo COLLATE SQL_Latin1_General_Cp1253_CI_AI) LIKE '%thap%'"

    # ===== Truy váº¥n chÃ­nh =====
    query = f"""
        SELECT ChanDoanID, BenhNhanID, TenBenhNhan, GioiTinh, Tuoi, TenBacSi, NgayChanDoan,
       BMI, HuyetApTamThu, HuyetApTamTruong, Cholesterol, DuongHuyet,
       HutThuoc, UongCon, TapTheDuc, NguyCo, LoiKhuyen

        FROM V_LichSuChanDoan
        {where_clause}
        ORDER BY NgayChanDoan {'DESC' if sort_order == 'desc' else 'ASC'}
    """

    cur.execute(query, params)
    records = cur.fetchall()
    conn.close()

    # âœ… Äáº¿m tá»•ng sá»‘ báº£n ghi
    total_records = len(records)

    # âœ… Highlight lá»i khuyÃªn
    try:
        from app import highlight_advice
        for r in records:
            if hasattr(r, "LoiKhuyen") and r.LoiKhuyen:
                r.LoiKhuyen = highlight_advice(r.LoiKhuyen)
    except Exception as e:
        print(f"âš ï¸ Lá»—i highlight: {e}")

    # ===== Danh sÃ¡ch lá»c =====
    doctors, patients = [], []

    if role == 'doctor':
        # Danh sÃ¡ch bá»‡nh nhÃ¢n mÃ  bÃ¡c sÄ© Ä‘Ã³ Ä‘Ã£ cháº©n Ä‘oÃ¡n
        conn2 = get_connection()
        cur2 = conn2.cursor()
        cur2.execute("""
            SELECT DISTINCT bn.ID, bn.HoTen 
            FROM ChanDoan cd 
            JOIN NguoiDung bn ON cd.BenhNhanID = bn.ID
            WHERE cd.BacSiID = ?
        """, (session['user_id'],))
        patients = cur2.fetchall()
        conn2.close()

    elif role == 'admin':
        # Danh sÃ¡ch bÃ¡c sÄ© vÃ  bá»‡nh nhÃ¢n cho admin
        conn2 = get_connection()
        cur2 = conn2.cursor()
        cur2.execute("SELECT ID, HoTen FROM NguoiDung WHERE Role='doctor'")
        doctors = cur2.fetchall()
        cur2.execute("SELECT ID, HoTen FROM NguoiDung WHERE Role='patient'")
        patients = cur2.fetchall()
        conn2.close()

    # ===== Render =====
    return render_template(
        'history.html',
        records=records,
        doctors=doctors,
        patients=patients,
        start_date=start_date,
        end_date=end_date,
        patient_id=patient_id,
        doctor_id=doctor_id,
        risk_filter=risk_filter,
        sort_order=sort_order,
        total_records=total_records
    )


# ==========================================
# ðŸ—‘ï¸ XÃ³a báº£n ghi cháº©n Ä‘oÃ¡n
# ==========================================
@app.route('/delete_history/<int:id>', methods=['POST'])
def delete_history(id):
    if 'user' not in session:
        return redirect(url_for('login'))

    role = session.get('role')
    if role not in ['doctor', 'admin','patient']:
        flash("âŒ Báº¡n khÃ´ng cÃ³ quyá»n xÃ³a báº£n ghi cháº©n Ä‘oÃ¡n.", "danger")
        return redirect(url_for('history'))

    conn = get_connection()
    cur = conn.cursor()
    try:
        # âœ… XÃ³a theo ID (khÃ³a chÃ­nh)
        cur.execute("DELETE FROM ChanDoan WHERE ID = ?", (id,))
        conn.commit()
        flash("ðŸ—‘ï¸ ÄÃ£ xÃ³a báº£n ghi cháº©n Ä‘oÃ¡n thÃ nh cÃ´ng!", "success")

    except Exception as e:
        conn.rollback()
        flash(f"âŒ Lá»—i khi xÃ³a báº£n ghi: {e}", "danger")

    finally:
        conn.close()

    return redirect(url_for('history'))

# ==========================================
# Chá»‰nh sá»­a lá»i khuyÃªn (chá»‰ dÃ nh cho bÃ¡c sÄ©)
# ==========================================
@app.route('/edit_advice/<int:id>', methods=['POST'])
def edit_advice(id):
    if 'user' not in session or session.get('role') != 'doctor':
        flash("âŒ Báº¡n khÃ´ng cÃ³ quyá»n chá»‰nh sá»­a lá»i khuyÃªn.", "danger")
        return redirect(url_for('login'))

    new_advice = request.form.get('loi_khuyen', '').strip()

    # ðŸ§¹ LÃ m sáº¡ch: loáº¡i bá» má»i tháº» HTML, style cÃ²n sÃ³t láº¡i
    import re
    from html import unescape
    clean_text = re.sub(r'<[^>]+>', '', new_advice)   # xÃ³a tháº» HTML
    clean_text = unescape(clean_text)                 # giáº£i mÃ£ HTML entities (&nbsp;)
    clean_text = re.sub(r'\s{2,}', ' ', clean_text)   # gá»™p khoáº£ng tráº¯ng

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            UPDATE ChanDoan
            SET LoiKhuyen = ?
            WHERE ID = ?
        """, (clean_text, id))
        conn.commit()
        flash("âœ… ÄÃ£ cáº­p nháº­t lá»i khuyÃªn cho bá»‡nh nhÃ¢n.", "success")

    except Exception as e:
        conn.rollback()
        flash(f"âŒ Lá»—i khi cáº­p nháº­t lá»i khuyÃªn: {e}", "danger")

    finally:
        conn.close()

    return redirect(url_for('history'))


# ==========================================
# Quáº£n lÃ½ tÃ i khoáº£n & há»“ sÆ¡ bá»‡nh nhÃ¢n (phiÃªn báº£n giá»›i háº¡n quyá»n)
# ==========================================
@app.route('/manage_accounts', methods=['GET', 'POST'])
def manage_accounts():
    # âœ… Chá»‰ cho phÃ©p bÃ¡c sÄ© truy cáº­p
    if 'user' not in session or session.get('role') != 'doctor':
        return redirect(url_for('login'))

    conn = get_connection()
    cur = conn.cursor()

    # ================================
    # âž• THÃŠM bá»‡nh nhÃ¢n má»›i
    # ================================
    if request.method == 'POST' and 'add_patient' in request.form:
        ho_ten = request.form.get('ho_ten')
        gioi_tinh = request.form.get('gioi_tinh')
        ngay_sinh = request.form.get('ngay_sinh')
        email = (request.form.get('email') or '').strip().lower()
        mat_khau = request.form.get('mat_khau')
        dien_thoai = request.form.get('dien_thoai')
        dia_chi = request.form.get('dia_chi')

        if not is_valid_gmail(email):
            flash("Vui lòng nhập Gmail hợp lệ (ví dụ: ten@gmail.com).", "warning")
        else:
            cur.execute("SELECT COUNT(*) FROM NguoiDung WHERE Email = ?", (email,))
            if cur.fetchone()[0] > 0:
                flash("Email này đã tồn tại trong hệ thống.", "warning")
            else:
                try:
                    cur.execute("""
                        INSERT INTO NguoiDung (HoTen, GioiTinh, NgaySinh, Email, MatKhau, DienThoai, DiaChi, Role)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 'patient')
                    """, (ho_ten, gioi_tinh, ngay_sinh, email, mat_khau, dien_thoai, dia_chi))
                    conn.commit()
                    flash("Đã thêm bệnh nhân mới thành công!", "success")
                except Exception as e:
                    conn.rollback()
                    flash(f"Lỗi khi thêm bệnh nhân: {e}", "danger")
            flash(f"âŒ Lá»—i khi thÃªm bá»‡nh nhÃ¢n: {e}", "danger")

    # ================================
    # ðŸ—‘ï¸ XÃ“A tÃ i khoáº£n bá»‡nh nhÃ¢n (chá»‰ náº¿u bÃ¡c sÄ© tá»«ng cháº©n Ä‘oÃ¡n)
    # ================================
    if request.method == 'POST' and 'delete_patient' in request.form:
        patient_id = int(request.form.get('id'))
        doctor_id = session['user_id']

        # Kiá»ƒm tra quyá»n trÆ°á»›c khi xÃ³a
        cur.execute("""
            SELECT COUNT(*) FROM ChanDoan 
            WHERE BacSiID=? AND BenhNhanID=?
        """, (doctor_id, patient_id))
        has_permission = cur.fetchone()[0] > 0

        if not has_permission:
            flash("ðŸš« Báº¡n khÃ´ng cÃ³ quyá»n xÃ³a bá»‡nh nhÃ¢n nÃ y (chÆ°a tá»«ng cháº©n Ä‘oÃ¡n).", "danger")
        else:
            try:
                cur.execute("DELETE FROM ChanDoan WHERE BenhNhanID=?", (patient_id,))
                cur.execute("DELETE FROM TinNhanAI WHERE BenhNhanID=?", (patient_id,))
                cur.execute("DELETE FROM NguoiDung WHERE ID=?", (patient_id,))
                conn.commit()
                flash("ðŸ—‘ï¸ ÄÃ£ xÃ³a tÃ i khoáº£n vÃ  toÃ n bá»™ lá»‹ch sá»­ cháº©n Ä‘oÃ¡n cá»§a bá»‡nh nhÃ¢n.", "success")
            except Exception as e:
                conn.rollback()
                flash(f"âŒ Lá»—i khi xÃ³a: {e}", "danger")

    # ================================
    # âœï¸ Cáº¬P NHáº¬T thÃ´ng tin bá»‡nh nhÃ¢n (chá»‰ náº¿u bÃ¡c sÄ© tá»«ng cháº©n Ä‘oÃ¡n)
    # ================================
    if request.method == 'POST' and 'update_patient' in request.form:
        patient_id = int(request.form.get('id'))
        doctor_id = session['user_id']

        cur.execute("""
            SELECT COUNT(*) FROM ChanDoan 
            WHERE BacSiID=? AND BenhNhanID=?
        """, (doctor_id, patient_id))
        has_permission = cur.fetchone()[0] > 0

        if not has_permission:
            flash("ðŸš« Báº¡n khÃ´ng cÃ³ quyá»n chá»‰nh sá»­a bá»‡nh nhÃ¢n nÃ y (chÆ°a tá»«ng cháº©n Ä‘oÃ¡n).", "danger")
        else:
            try:
                cur.execute("""
                    UPDATE NguoiDung
                    SET HoTen=?, GioiTinh=?, NgaySinh=?, DienThoai=?, DiaChi=?
                    WHERE ID=?
                """, (
                    request.form.get('ho_ten'),
                    request.form.get('gioi_tinh'),
                    request.form.get('ngay_sinh'),
                    request.form.get('dien_thoai'),
                    request.form.get('dia_chi'),
                    patient_id
                ))
                conn.commit()
                flash("âœ… ÄÃ£ cáº­p nháº­t thÃ´ng tin bá»‡nh nhÃ¢n.", "success")
            except Exception as e:
                conn.rollback()
                flash(f"âŒ Lá»—i khi cáº­p nháº­t: {e}", "danger")

    # ================================
    # ðŸ”Ž TÃŒM KIáº¾M bá»‡nh nhÃ¢n
    # ================================
    search = request.args.get('search', '').strip()

    if search:
        cur.execute("""
            SELECT ID, HoTen, Email, GioiTinh, NgaySinh, DienThoai, DiaChi
            FROM NguoiDung
            WHERE Role='patient' AND (HoTen LIKE ? OR Email LIKE ?)
            ORDER BY HoTen
        """, (f"%{search}%", f"%{search}%"))
    else:
        cur.execute("""
            SELECT ID, HoTen, Email, GioiTinh, NgaySinh, DienThoai, DiaChi
            FROM NguoiDung
            WHERE Role='patient'
            ORDER BY HoTen
        """)

    raw_patients = cur.fetchall()

    # ================================
    # ðŸ” Láº¥y danh sÃ¡ch bá»‡nh nhÃ¢n bÃ¡c sÄ© tá»«ng cháº©n Ä‘oÃ¡n
    # ================================
    cur.execute("""
        SELECT DISTINCT BenhNhanID FROM ChanDoan WHERE BacSiID=?
    """, (session['user_id'],))
    my_patients = {r.BenhNhanID for r in cur.fetchall()}

    # ================================
    # Xá»¬ LÃ dá»¯ liá»‡u hiá»ƒn thá»‹
    # ================================
    patients = []
    for p in raw_patients:
        if p.NgaySinh and hasattr(p.NgaySinh, "strftime"):
            ngay_sinh_str = p.NgaySinh.strftime("%d/%m/%Y")
            ngay_sinh_val = p.NgaySinh.strftime("%Y-%m-%d")
        else:
            ngay_sinh_str = p.NgaySinh if p.NgaySinh else "â€”"
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

    # âœ… Truyá»n thÃªm danh sÃ¡ch quyá»n my_patients sang template
    return render_template(
        'manage_accounts.html',
        patients=patients,
        search=search,
        my_patients=my_patients
    )

from flask import flash
from werkzeug.security import check_password_hash, generate_password_hash

import re
from flask import jsonify

# ==========================================
# ðŸ” Äá»•i máº­t kháº©u (xá»­ lÃ½ AJAX)
# ==========================================
@app.route('/change_password', methods=['POST'])
def change_password():
    if 'user' not in session:
        return jsonify({"success": False, "message": "Vui lÃ²ng Ä‘Äƒng nháº­p láº¡i."}), 403

    old_pw = request.form.get('old_password')
    new_pw = request.form.get('new_password')
    confirm_pw = request.form.get('confirm_password')

    if not old_pw or not new_pw or not confirm_pw:
        return jsonify({"success": False, "message": "Vui lÃ²ng nháº­p Ä‘áº§y Ä‘á»§ thÃ´ng tin."})

    if new_pw != confirm_pw:
        return jsonify({"success": False, "message": "Máº­t kháº©u xÃ¡c nháº­n khÃ´ng khá»›p."})

    # ðŸ§© Kiá»ƒm tra Ä‘á»™ máº¡nh máº­t kháº©u (Ã­t nháº¥t 8 kÃ½ tá»±, cÃ³ hoa, sá»‘, Ä‘áº·c biá»‡t)
    if not re.match(r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$', new_pw):
        return jsonify({
            "success": False,
            "message": "Máº­t kháº©u pháº£i â‰¥8 kÃ½ tá»±, chá»©a Ã­t nháº¥t 1 chá»¯ hoa, 1 sá»‘ vÃ  1 kÃ½ tá»± Ä‘áº·c biá»‡t."
        })

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT MatKhau FROM NguoiDung WHERE ID=?", (session['user_id'],))
    row = cur.fetchone()

    if not row or row.MatKhau != old_pw:
        conn.close()
        return jsonify({"success": False, "message": "Máº­t kháº©u cÅ© khÃ´ng chÃ­nh xÃ¡c."})

    cur.execute("UPDATE NguoiDung SET MatKhau=? WHERE ID=?", (new_pw, session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Äá»•i máº­t kháº©u thÃ nh cÃ´ng!"})


# ==========================================
# Há»“ sÆ¡ cÃ¡ nhÃ¢n
# ==========================================
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    cur = conn.cursor()

    # --- Khi ngÆ°á»i dÃ¹ng cáº­p nháº­t há»“ sÆ¡ ---
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

        # LÆ°u thá»i gian cáº­p nháº­t táº¡m vÃ o session
        from datetime import datetime
        update_time = datetime.now().strftime("%d/%m/%Y %H:%M")
        if 'timeline' not in session:
            session['timeline'] = []
        session['timeline'].insert(0, f"Cáº­p nháº­t há»“ sÆ¡ - {update_time}")

        flash("Cáº­p nháº­t há»“ sÆ¡ thÃ nh cÃ´ng!", "success")

    # --- Láº¥y thÃ´ng tin ngÆ°á»i dÃ¹ng (bao gá»“m ngÃ y táº¡o tÃ i khoáº£n) ---
    cur.execute("""
        SELECT HoTen, Email, Role, DienThoai, NgaySinh, GioiTinh, DiaChi, MatKhau, NgayTao
        FROM NguoiDung WHERE ID=?
    """, (session['user_id'],))
    user_info = cur.fetchone()
    can_change_password = False
    if user_info:
        mat_khau_val = getattr(user_info, 'MatKhau', None)
        if isinstance(mat_khau_val, str):
            mat_khau_val = mat_khau_val.strip()
        can_change_password = bool(mat_khau_val)
    conn.close()

    # --- Chuáº©n bá»‹ timeline hiá»ƒn thá»‹ ---
    timeline = []
    if user_info and user_info[-1]:
        # user_info[-1] = NgayTao
        created_at = user_info[-1].strftime("%d/%m/%Y %H:%M")
        timeline.append(f"Táº¡o tÃ i khoáº£n - {created_at}")
    if 'timeline' in session:
        timeline = session['timeline'] + timeline

    return render_template(
        'profile.html',
        user_info=user_info,
        user_timeline=timeline,
        can_change_password=can_change_password
    )

# ==========================================
# ðŸ“¤ Xuáº¥t bÃ¡o cÃ¡o káº¿t quáº£ cháº©n Ä‘oÃ¡n ra Excel 
# ==========================================
@app.route('/export_diagnosis', methods=['POST'])
def export_diagnosis():
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as ExcelImage
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from io import BytesIO
    from flask import send_file
    from datetime import datetime
    import os, re

    # ===== Dá»¯ liá»‡u tá»« form =====
    data = {key: request.form.get(key, '') for key in [
        'age', 'gender', 'bmi', 'systolic', 'diastolic', 'cholesterol',
        'glucose', 'smoking', 'alcohol', 'exercise',
        'risk_percent', 'risk_level', 'ai_advice', 'shap_file', 'benhnhan_id'
    ]}

    # ===== Láº¥y tÃªn ngÆ°á»i Ä‘Äƒng nháº­p & vai trÃ² =====
    user_name = session.get('user', 'NgÆ°á»i dÃ¹ng')
    user_role = session.get('role', 'patient')

    # ===== XÃ¡c Ä‘á»‹nh tÃªn bá»‡nh nhÃ¢n vÃ  bÃ¡c sÄ© =====
    patient_name = None
    doctor_name = None

    if user_role == 'doctor':
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT HoTen FROM NguoiDung WHERE ID = ?", data.get('benhnhan_id'))
        row = cur.fetchone()
        conn.close()
        patient_name = row[0] if row else "KhÃ´ng xÃ¡c Ä‘á»‹nh"
        doctor_name = user_name
    else:
        patient_name = user_name
        doctor_name = "â€”"

    # ===== Táº¡o workbook =====
    wb = Workbook()
    ws = wb.active
    ws.title = "BÃ¡o cÃ¡o cháº©n Ä‘oÃ¡n"

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

    # ===== TiÃªu Ä‘á» =====
    ws.merge_cells("A1:E1")
    ws["A1"] = "BÃO CÃO Káº¾T QUáº¢ CHáº¨N ÄOÃN TIM Máº CH"
    ws["A1"].font = title_font
    ws["A1"].alignment = center
    ws.append([])

    # ===== I. ThÃ´ng tin chung =====
    ws.merge_cells("A3:E3")
    ws["A3"] = "I. THÃ”NG TIN CHUNG"
    ws["A3"].font = section_font
    ws["A3"].alignment = left

    ws.append(["TÃªn bá»‡nh nhÃ¢n", patient_name])
    ws.append(["BÃ¡c sÄ© cháº©n Ä‘oÃ¡n", doctor_name])
    ws.append(["NgÃ y táº¡o bÃ¡o cÃ¡o", datetime.now().strftime("%d/%m/%Y %H:%M")])
    ws.append([])

    # ===== II. Dá»¯ liá»‡u Ä‘áº§u vÃ o =====
    ws.merge_cells("A7:E7")
    ws["A7"] = "II. Dá»® LIá»†U Äáº¦U VÃ€O"
    ws["A7"].font = section_font
    ws["A7"].alignment = left

    ws.append(["Thuá»™c tÃ­nh", "GiÃ¡ trá»‹", "Thuá»™c tÃ­nh", "GiÃ¡ trá»‹"])
    for cell in ws[8]:
        cell.font = header_font
        cell.fill = fill_header
        cell.border = border
        cell.alignment = center

    input_data = [
        ["Tuá»•i", data['age'], "Giá»›i tÃ­nh", data['gender']],
        ["BMI", data['bmi'], "Huyáº¿t Ã¡p (HATT/HATTr)", f"{data['systolic']}/{data['diastolic']}"],
        ["Cholesterol", data['cholesterol'], "ÄÆ°á»ng huyáº¿t", data['glucose']],
        ["HÃºt thuá»‘c", "CÃ³" if data['smoking']=="yes" else "KhÃ´ng", "RÆ°á»£u/Bia", "CÃ³" if data['alcohol']=="yes" else "KhÃ´ng"],
        ["Táº­p thá»ƒ dá»¥c", "CÃ³" if data['exercise']=="yes" else "KhÃ´ng", "", ""]
    ]
    for row in input_data:
        ws.append(row)
        for cell in ws[ws.max_row]:
            cell.font = normal_font
            cell.border = border
            cell.alignment = left

    ws.append([])

        # ===== III. Káº¿t quáº£ cháº©n Ä‘oÃ¡n =====
    ws.merge_cells(f"A{ws.max_row+1}:E{ws.max_row+1}")
    ws[f"A{ws.max_row}"] = "III. Káº¾T QUáº¢ CHáº¨N ÄOÃN"
    ws[f"A{ws.max_row}"].font = section_font
    ws[f"A{ws.max_row}"].alignment = left

    ws.append(["Nguy cÆ¡", "Tá»‰ lá»‡ (%)", "ÄÃ¡nh giÃ¡", ""])
    for cell in ws[ws.max_row]:
        cell.font = header_font
        cell.fill = fill_header
        cell.border = border
        cell.alignment = center

    ws.append([
        "Cao" if data['risk_level'] == 'high' else "Tháº¥p",
        data['risk_percent'] + "%",
        "âš ï¸ Cáº§n theo dÃµi" if data['risk_level'] == 'high' else "âœ… á»”n Ä‘á»‹nh",
        ""
    ])
    for cell in ws[ws.max_row]:
        cell.font = normal_font
        cell.border = border
        cell.alignment = center
        cell.fill = fill_high if data['risk_level'] == 'high' else fill_low

    ws.append([])

    import re
    from html import unescape

    # ===== LÃ m sáº¡ch vÃ  Ä‘á»‹nh dáº¡ng lá»i khuyÃªn =====
    advice_raw = data.get('ai_advice') or "ChÆ°a cÃ³ lá»i khuyÃªn tá»« AI."

    # âœ… Bá» toÃ n bá»™ tháº» HTML & thuá»™c tÃ­nh style
    advice_text = re.sub(r'style="[^"]*"', '', advice_raw)     # xÃ³a thuá»™c tÃ­nh style
    advice_text = re.sub(r'<[^>]+>', '', advice_text)          # xÃ³a tháº» HTML cÃ²n láº¡i
    advice_text = unescape(advice_text)                        # giáº£i mÃ£ HTML entity (&nbsp;,...)
    advice_text = re.sub(r'\s*\n\s*', '\n', advice_text.strip())
    advice_text = re.sub(r'\s{2,}', ' ', advice_text)

    # âœ… Tá»± Ä‘á»™ng ngáº¯t dÃ²ng sau dáº¥u cháº¥m (khi sau Ä‘Ã³ lÃ  chá»¯ in hoa hoáº·c tiáº¿ng Viá»‡t cÃ³ dáº¥u)
    advice_text = re.sub(r'\.\s*(?=[A-ZÃ€-á»¸])', '.\n', advice_text)

    # âœ… Ngáº¯t dÃ²ng trÆ°á»›c cÃ¡c cá»¥m tá»« nhÆ° â€œLá»i khuyÃªnâ€, â€œKhuyáº¿n nghá»‹â€, â€œTÃ³m láº¡iâ€
    advice_text = re.sub(r'(?=\b(Lá»i khuyÃªn|Khuyáº¿n nghá»‹|TÃ³m láº¡i)\b)', '\n', advice_text)

    # âœ… Loáº¡i bá» dÃ²ng trá»‘ng dÆ°
    advice_text = re.sub(r'\n{2,}', '\n', advice_text).strip()

    # âœ… Ghi ra Excel (xuá»‘ng dÃ²ng, cÄƒn Ä‘á»u 2 bÃªn)
    start_row = ws.max_row + 1
    end_row = start_row + 8
    ws.merge_cells(f"A{start_row}:E{end_row}")
    cell = ws[f"A{start_row}"]
    cell.value = advice_text
    cell.alignment = Alignment(horizontal="justify", vertical="top", wrap_text=True)
    cell.font = advice_font
    cell.border = border
    cell.fill = fill_sub

    # ===== V. Biá»ƒu Ä‘á»“ SHAP =====
    shap_path = os.path.join(app.root_path, 'static', 'images', data['shap_file']) if data['shap_file'] else None
    if shap_path and os.path.exists(shap_path):
        ws.merge_cells(f"A{ws.max_row+1}:E{ws.max_row+1}")
        ws[f"A{ws.max_row}"] = "V. GIáº¢I THÃCH Káº¾T QUáº¢ Báº°NG BIá»‚U Äá»’ SHAP"
        ws[f"A{ws.max_row}"].font = section_font
        ws[f"A{ws.max_row}"].alignment = left
        try:
            img = ExcelImage(shap_path)
            img.width = 520
            img.height = 320
            ws.add_image(img, f"A{ws.max_row+1}")
        except Exception as e:
            ws.append([f"Lá»—i khi chÃ¨n hÃ¬nh: {e}"])

    # ===== Footer =====
    ws.append([])
    ws.merge_cells(f"A{ws.max_row}:E{ws.max_row}")
    ws[f"A{ws.max_row}"] = f"ðŸ“… BÃ¡o cÃ¡o Ä‘Æ°á»£c táº¡o bá»Ÿi: {doctor_name or user_name} â€” {datetime.now().strftime('%H:%M, %d/%m/%Y')}"
    ws[f"A{ws.max_row}"].alignment = center
    ws[f"A{ws.max_row}"].font = Font(size=10, italic=True, color="777777")

    # ===== CÄƒn chá»‰nh Ä‘á»™ rá»™ng =====
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 20
    ws.column_dimensions["C"].width = 25
    ws.column_dimensions["D"].width = 25
    ws.column_dimensions["E"].width = 10

    # ===== Xuáº¥t file =====
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
# ðŸšª ÄÄƒng xuáº¥t
# ==========================================
@app.route('/logout')
def logout():
    if 'user' in session:
        name = session.get('user', 'NgÆ°á»i dÃ¹ng')
        session.clear()
        flash(f"ðŸ‘‹ {name}, báº¡n Ä‘Ã£ Ä‘Äƒng xuáº¥t khá»i há»‡ thá»‘ng thÃ nh cÃ´ng!", "success")
    else:
        flash("âš ï¸ Báº¡n chÆ°a Ä‘Äƒng nháº­p!", "warning")
    return redirect(url_for('login'))
# =========================================================
# ðŸ“Š DASHBOARD THá»NG KÃŠ (Admin - Báº£n nÃ¢ng cáº¥p chuyÃªn sÃ¢u)
# =========================================================
@app.route('/admin/dashboard')
def admin_dashboard():
    # --- Kiá»ƒm tra quyá»n truy cáº­p ---
    if 'user' not in session or session.get('role') != 'admin':
        flash("Báº¡n khÃ´ng cÃ³ quyá»n truy cáº­p trang nÃ y!", "danger")
        return redirect(url_for('login'))

    conn = get_connection()
    cur = conn.cursor()

    # ========================== 1ï¸âƒ£ Tá»”NG QUAN ==========================
    cur.execute("SELECT COUNT(*) FROM NguoiDung WHERE Role='doctor'")
    total_doctors = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM NguoiDung WHERE Role='patient'")
    total_patients = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM ChanDoan")
    total_diagnoses = cur.fetchone()[0]

    # ========================== 2ï¸âƒ£ XU HÆ¯á»šNG CHáº¨N ÄOÃN ==========================
    cur.execute("""
        SELECT FORMAT(NgayChanDoan, 'MM-yyyy') AS Thang,
               COUNT(*) AS SoLuong,
               SUM(CASE WHEN LOWER(NguyCo) LIKE '%cao%' THEN 1 ELSE 0 END) AS SoCao
        FROM ChanDoan
        GROUP BY FORMAT(NgayChanDoan, 'MM-yyyy')
        ORDER BY MIN(NgayChanDoan)
    """)
    monthly = cur.fetchall()
    months = [row.Thang for row in monthly]
    counts = [row.SoLuong for row in monthly]
    high_risk = [row.SoCao for row in monthly]

    # ========================== 3ï¸âƒ£ Tá»¶ Lá»† NGUY CÆ  ==========================
    cur.execute("""
        SELECT NguyCo, COUNT(*) AS SoLuong
        FROM ChanDoan
        GROUP BY NguyCo
    """)
    risk_data = cur.fetchall()
    risk_labels = [row.NguyCo for row in risk_data]
    risk_values = [row.SoLuong for row in risk_data]

    # ========================== 4ï¸âƒ£ TOP BÃC SÄ¨ ==========================
    cur.execute("""
        SELECT TOP 5 bs.HoTen, COUNT(cd.ID) AS SoCa,
               SUM(CASE WHEN cd.NguyCo LIKE '%cao%' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS TyLeCao
        FROM ChanDoan cd
        JOIN NguoiDung bs ON cd.BacSiID = bs.ID
        GROUP BY bs.HoTen
        ORDER BY SoCa DESC
    """)
    top_doctors = cur.fetchall()
    top_names = [row.HoTen for row in top_doctors]
    top_counts = [row.SoCa for row in top_doctors]
    top_rates = [round(row.TyLeCao, 1) for row in top_doctors]

    # ========================== 5ï¸âƒ£ TRUNG BÃŒNH CHá»ˆ Sá» Y KHOA ==========================
    cur.execute("""
        SELECT 
            ROUND(AVG(BMI), 1) AS AvgBMI,
            ROUND(AVG(HuyetApTamThu), 0) AS AvgHATT,
            ROUND(AVG(HuyetApTamTruong), 0) AS AvgHATTr,
            SUM(CASE WHEN HutThuoc=1 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS SmokePercent,
            SUM(CASE WHEN UongCon=1 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS AlcoPercent,
            SUM(CASE WHEN TapTheDuc=1 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS ActivePercent
        FROM ChanDoan
    """)
    row = cur.fetchone()
    avg_bmi = row.AvgBMI or 0
    avg_systolic = row.AvgHATT or 0
    avg_diastolic = row.AvgHATTr or 0
    smoke_percent = round(row.SmokePercent or 0, 1)
    alco_percent = round(row.AlcoPercent or 0, 1)
    active_percent = round(row.ActivePercent or 0, 1)

    # ========================== 6ï¸âƒ£ HIá»†U SUáº¤T BÃC SÄ¨ ==========================
    cur.execute("""
        SELECT ND.HoTen AS BacSi,
               COUNT(CD.ID) AS SoCa,
               SUM(CASE WHEN CD.NguyCo LIKE '%cao%' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS TyLeCao
        FROM ChanDoan CD
        JOIN NguoiDung ND ON CD.BacSiID = ND.ID
        GROUP BY ND.HoTen
        ORDER BY SoCa DESC
    """)
    perf_rows = cur.fetchall()
    perf_names = [r.BacSi for r in perf_rows]
    perf_cases = [r.SoCa for r in perf_rows]
    perf_rate = [round(r.TyLeCao or 0, 1) for r in perf_rows]

    # ========================== 7ï¸âƒ£ Tá»”NG Sá» Bá»†NH NHÃ‚N CÃ“ CHáº¨N ÄOÃN ==========================
    cur.execute("SELECT COUNT(DISTINCT BenhNhanID) FROM ChanDoan")
    diagnosed_patients = cur.fetchone()[0]

    conn.close()

    return render_template(
        'admin_dashboard.html',
        total_doctors=total_doctors,
        total_patients=total_patients,
        total_diagnoses=total_diagnoses,
        diagnosed_patients=diagnosed_patients,
        months=months,
        counts=counts,
        high_risk=high_risk,
        risk_labels=risk_labels,
        risk_values=risk_values,
        top_names=top_names,
        top_counts=top_counts,
        top_rates=top_rates,
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
# ðŸ§‘â€âš•ï¸ Quáº£n lÃ½ ngÆ°á»i dÃ¹ng (BÃ¡c sÄ© / Bá»‡nh nhÃ¢n) â€” Admin
# =========================================================
@app.route('/admin/manage_users', methods=['GET', 'POST'])
def admin_manage_users():
    if 'user' not in session or session.get('role') != 'admin':
        flash("âŒ Báº¡n khÃ´ng cÃ³ quyá»n truy cáº­p trang nÃ y!", "danger")
        return redirect(url_for('login'))

    import datetime
    conn = get_connection()
    cur = conn.cursor()

    # XÃ¡c Ä‘á»‹nh loáº¡i ngÆ°á»i dÃ¹ng Ä‘ang quáº£n lÃ½
    role_type = request.args.get('type', 'doctor')  # máº·c Ä‘á»‹nh lÃ  doctor
    if request.method == 'POST' and 'add_user' in request.form:
        ho_ten = request.form.get('ho_ten', '').strip()
        email = (request.form.get('email', '').strip().lower())
        mat_khau = request.form.get('mat_khau', '').strip()
        gioi_tinh = request.form.get('gioi_tinh')
        ngay_sinh = request.form.get('ngay_sinh') or None
        dien_thoai = request.form.get('dien_thoai')
        dia_chi = request.form.get('dia_chi')

        if not is_valid_gmail(email):
            flash("Vui lòng nhập Gmail hợp lệ (ví dụ: ten@gmail.com).", "warning")
        else:
            cur.execute("SELECT COUNT(*) FROM NguoiDung WHERE Email = ?", (email,))
            if cur.fetchone()[0] > 0:
                flash("Email này đã tồn tại trong hệ thống.", "warning")
            else:
                cur.execute("""
                    INSERT INTO NguoiDung (HoTen, Email, MatKhau, Role, NgaySinh, GioiTinh, DienThoai, DiaChi)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (ho_ten, email, mat_khau, role_type, ngay_sinh, gioi_tinh, dien_thoai, dia_chi))
                conn.commit()
                flash(f"Đã thêm {title_map[role_type]} mới thành công!", "success")
    # ===================================================
    # âœï¸ Sá»¬A NGÆ¯á»œI DÃ™NG
    # ===================================================
    elif request.method == 'POST' and 'edit_user' in request.form:
        id = request.form.get('id')
        ho_ten = request.form.get('ho_ten', '').strip()
        gioi_tinh = request.form.get('gioi_tinh')
        ngay_sinh = request.form.get('ngay_sinh') or None
        email = request.form.get('email', '').strip().lower()
        mat_khau = request.form.get('mat_khau', '').strip()
        dien_thoai = request.form.get('dien_thoai')
        dia_chi = request.form.get('dia_chi')

        if not mat_khau:
            cur.execute("""
                UPDATE NguoiDung
                SET HoTen=?, GioiTinh=?, NgaySinh=?, Email=?, DienThoai=?, DiaChi=?
                WHERE ID=? AND Role=?
            """, (ho_ten, gioi_tinh, ngay_sinh, email, dien_thoai, dia_chi, id, role_type))
        else:
            cur.execute("""
                UPDATE NguoiDung
                SET HoTen=?, GioiTinh=?, NgaySinh=?, Email=?, MatKhau=?, DienThoai=?, DiaChi=?
                WHERE ID=? AND Role=?
            """, (ho_ten, gioi_tinh, ngay_sinh, email, mat_khau, dien_thoai, dia_chi, id, role_type))
        conn.commit()
        flash(f"âœï¸ Cáº­p nháº­t thÃ´ng tin {title_map[role_type]} thÃ nh cÃ´ng!", "success")

    # ===================================================
    # ðŸ—‘ï¸ XÃ“A NGÆ¯á»œI DÃ™NG
    # ===================================================
    elif request.method == 'POST' and 'delete_user' in request.form:
        user_id = int(request.form.get('id'))
        try:
            if role_type == 'patient':
                cur.execute("DELETE FROM ChanDoan WHERE BenhNhanID=?", (user_id,))
                cur.execute("DELETE FROM TinNhanAI WHERE BenhNhanID=?", (user_id,))

            cur.execute("DELETE FROM NguoiDung WHERE ID=? AND Role=?", (user_id, role_type))
            conn.commit()
            flash(f"ÄÃ£ xÃ³a {title_map[role_type]} khá»i há»‡ thá»‘ng!", "success")
        except Exception as e:
            conn.rollback()
            flash(f"Lá»—i khi xÃ³a tÃ i khoáº£n: {e}", "danger")

    # ===================================================
    # ðŸ“‹ DANH SÃCH NGÆ¯á»œI DÃ™NG
    # ===================================================
    cur.execute(f"""
        SELECT ID, HoTen, Email, GioiTinh, NgaySinh, DienThoai, DiaChi, NgayTao
        FROM NguoiDung
        WHERE Role=?
        ORDER BY NgayTao DESC
    """, (role_type,))
    users = cur.fetchall()

    # Chuyá»ƒn ngÃ y sang kiá»ƒu datetime
    for u in users:
        if hasattr(u, 'NgaySinh') and isinstance(u.NgaySinh, str):
            try:
                u.NgaySinh = datetime.datetime.strptime(u.NgaySinh.split(" ")[0], "%Y-%m-%d")
            except:
                u.NgaySinh = None
        if hasattr(u, 'NgayTao') and isinstance(u.NgayTao, str):
            try:
                u.NgayTao = datetime.datetime.strptime(u.NgayTao.split(" ")[0], "%Y-%m-%d")
            except:
                u.NgayTao = None

    conn.close()

    return render_template('admin_users.html',
                           users=users,
                           role_type=role_type,
                           page_title=page_title)


# ==========================================
# ðŸ“Š XUáº¤T FILE EXCEL THá»NG KÃŠ Há»† THá»NG - NÃ¢ng cáº¥p chuyÃªn nghiá»‡p
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

    # =============================== #
    # ðŸ“¥ Láº¤Y Dá»® LIá»†U Tá»ª DATABASE
    # =============================== #
    cur.execute("SELECT COUNT(*) FROM NguoiDung WHERE Role='doctor'")
    total_doctors = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM NguoiDung WHERE Role='patient'")
    total_patients = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM ChanDoan")
    total_diagnoses = cur.fetchone()[0]

    cur.execute("""
        SELECT NguyCo, COUNT(*) AS SoLuong
        FROM ChanDoan
        GROUP BY NguyCo
    """)
    risk_data = cur.fetchall()

    cur.execute("""
        SELECT TOP 5 bs.HoTen, COUNT(cd.ID) AS SoCa
        FROM ChanDoan cd
        JOIN NguoiDung bs ON cd.BacSiID = bs.ID
        GROUP BY bs.HoTen
        ORDER BY SoCa DESC
    """)
    top_doctors = cur.fetchall()

    cur.execute("""
        SELECT ND.HoTen AS BacSi,
               COUNT(CD.ID) AS SoCa,
               SUM(CASE WHEN CD.NguyCo LIKE '%cao%' THEN 1 ELSE 0 END)*100.0/COUNT(*) AS TyLeCao
        FROM ChanDoan CD
        JOIN NguoiDung ND ON CD.BacSiID = ND.ID
        GROUP BY ND.HoTen
        ORDER BY SoCa DESC
    """)
    perf_rows = cur.fetchall()
    conn.close()

    # =============================== #
    # ðŸ“˜ Táº O FILE EXCEL
    # =============================== #
    wb = Workbook()
    ws = wb.active
    ws.title = "Tá»•ng quan há»‡ thá»‘ng"

    # --- STYLE ---
    title_font = Font(size=16, bold=True, color="1F4E78")
    header_font = Font(size=12, bold=True, color="FFFFFF")
    fill_blue = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    fill_gray = PatternFill(start_color="E9ECEF", end_color="E9ECEF", fill_type="solid")
    align_center = Alignment(horizontal="center", vertical="center")
    border = Border(
        left=Side(style="thin", color="999999"),
        right=Side(style="thin", color="999999"),
        top=Side(style="thin", color="999999"),
        bottom=Side(style="thin", color="999999")
    )

    # =============================== #
    # ðŸ“„ SHEET 1: Tá»”NG QUAN
    # =============================== #
    ws.merge_cells("A1:E1")
    ws["A1"] = "BÃO CÃO THá»NG KÃŠ Há»† THá»NG CHáº¨N ÄOÃN TIM Máº CH"
    ws["A1"].font = title_font
    ws["A1"].alignment = align_center

    ws.append([])
    ws.append(["NgÃ y xuáº¥t bÃ¡o cÃ¡o:", datetime.now().strftime("%d/%m/%Y %H:%M")])
    ws.append(["NgÆ°á»i xuáº¥t:", session.get('user', 'Quáº£n trá»‹ viÃªn')])
    ws.append([])
    ws.append(["ðŸ“Š Chá»‰ sá»‘ tá»•ng quan"])
    ws.append(["Tá»•ng sá»‘ bÃ¡c sÄ©", total_doctors])
    ws.append(["Tá»•ng sá»‘ bá»‡nh nhÃ¢n", total_patients])
    ws.append(["Tá»•ng lÆ°á»£t cháº©n Ä‘oÃ¡n", total_diagnoses])
    ws.append([])

    ws.append(["ðŸ† Top 5 bÃ¡c sÄ© cÃ³ sá»‘ ca cháº©n Ä‘oÃ¡n nhiá»u nháº¥t"])
    ws.append(["TÃªn bÃ¡c sÄ©", "Sá»‘ ca"])
    for cell in ws[ws.max_row]:
        cell.font = header_font
        cell.fill = fill_blue
        cell.alignment = align_center
        cell.border = border

    for idx, d in enumerate(top_doctors, start=1):
        ws.append([d.HoTen, d.SoCa])
        for cell in ws[ws.max_row]:
            cell.border = border
            if idx % 2 == 0:
                cell.fill = fill_gray

    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 20

    # =============================== #
    # ðŸ“Š SHEET 2: Tá»¶ Lá»† BÃC SÄ¨ / Bá»†NH NHÃ‚N
    # =============================== #
    ws2 = wb.create_sheet("BÃ¡c sÄ© - Bá»‡nh nhÃ¢n")
    ws2.append(["Loáº¡i tÃ i khoáº£n", "Sá»‘ lÆ°á»£ng"])
    ws2.append(["BÃ¡c sÄ©", total_doctors])
    ws2.append(["Bá»‡nh nhÃ¢n", total_patients])

    for cell in ws2[1]:
        cell.font = header_font
        cell.fill = fill_blue
        cell.alignment = align_center
        cell.border = border
    for row in ws2.iter_rows(min_row=2, max_col=2):
        for c in row:
            c.border = border
            c.alignment = align_center

    from openpyxl.chart.label import DataLabelList

    pie = PieChart()
    pie.title = "Tá»· lá»‡ BÃ¡c sÄ© / Bá»‡nh nhÃ¢n"
    data = Reference(ws2, min_col=2, min_row=1, max_row=3)
    labels = Reference(ws2, min_col=1, min_row=2, max_row=3)
    pie.add_data(data, titles_from_data=True)
    pie.set_categories(labels)

    # âœ… Hiá»ƒn thá»‹ giÃ¡ trá»‹ + pháº§n trÄƒm + tÃªn
    pie.dLbls = DataLabelList()
    pie.dLbls.showVal = True
    pie.dLbls.showPercent = True
    pie.dLbls.showCatName = True

    ws2.add_chart(pie, "D5")


    # =============================== #
    # ðŸ“Š SHEET 3: Tá»¶ Lá»† NGUY CÆ 
    # =============================== #
    ws3 = wb.create_sheet("Nguy cÆ¡ cao - tháº¥p")
    ws3.append(["Má»©c nguy cÆ¡", "Sá»‘ lÆ°á»£ng"])
    for r in risk_data:
        ws3.append([r.NguyCo, r.SoLuong])

    for cell in ws3[1]:
        cell.font = header_font
        cell.fill = fill_blue
        cell.alignment = align_center
        cell.border = border

    bar = BarChart()
    bar.title = "Tá»· lá»‡ nguy cÆ¡ cao / tháº¥p"
    data = Reference(ws3, min_col=2, min_row=1, max_row=ws3.max_row)
    cats = Reference(ws3, min_col=1, min_row=2, max_row=ws3.max_row)
    bar.add_data(data, titles_from_data=True)
    bar.set_categories(cats)
    bar.y_axis.title = "Sá»‘ lÆ°á»£ng"
    ws3.add_chart(bar, "E5")

    # =============================== #
    # ðŸ“Š SHEET 4: HIá»†U SUáº¤T BÃC SÄ¨
    # =============================== #
    ws4 = wb.create_sheet("Hiá»‡u suáº¥t bÃ¡c sÄ©")
    ws4.append(["BÃ¡c sÄ©", "Sá»‘ ca", "Tá»· lá»‡ nguy cÆ¡ cao (%)"])
    for p in perf_rows:
        ws4.append([p.BacSi, p.SoCa, round(p.TyLeCao or 0, 1)])

    for cell in ws4[1]:
        cell.font = header_font
        cell.fill = fill_blue
        cell.alignment = align_center
        cell.border = border

    for row in ws4.iter_rows(min_row=2, max_col=3):
        for cell in row:
            cell.border = border
            if row[0].row % 2 == 0:
                cell.fill = fill_gray
            cell.alignment = align_center

    # --- Biá»ƒu Ä‘á»“ káº¿t há»£p ---
    chart = BarChart()
    chart.title = "Hiá»‡u suáº¥t & Tá»· lá»‡ nguy cÆ¡ cao cá»§a bÃ¡c sÄ©"
    chart.y_axis.title = "Sá»‘ ca"
    data_bar = Reference(ws4, min_col=2, min_row=1, max_row=ws4.max_row)
    cats = Reference(ws4, min_col=1, min_row=2, max_row=ws4.max_row)
    chart.add_data(data_bar, titles_from_data=True)
    chart.set_categories(cats)

    line = LineChart()
    data_line = Reference(ws4, min_col=3, min_row=1, max_row=ws4.max_row)
    line.add_data(data_line, titles_from_data=True)
    line.y_axis.title = "Tá»· lá»‡ (%)"
    line.y_axis.axId = 200
    chart.y_axis.crosses = "max"
    chart += line
    ws4.add_chart(chart, "E5")

    # =============================== #
    # ðŸ“Š SHEET 5: GHI CHÃš & CHá»® KÃ
    # =============================== #
    ws5 = wb.create_sheet("Ghi chÃº & Chá»¯ kÃ½")
    ws5["A1"] = "Ghi chÃº:"
    ws5["A2"] = "â€¢ BÃ¡o cÃ¡o Ä‘Æ°á»£c xuáº¥t tá»± Ä‘á»™ng tá»« há»‡ thá»‘ng CVD-App."
    ws5["A3"] = "â€¢ Dá»¯ liá»‡u cáº­p nháº­t Ä‘áº¿n thá»i Ä‘iá»ƒm xuáº¥t file."
    ws5["A5"] = "NgÆ°á»i láº­p bÃ¡o cÃ¡o:"
    ws5["A6"] = session.get('user', 'Quáº£n trá»‹ viÃªn')
    ws5["A8"] = "Chá»¯ kÃ½:"
    ws5["A9"] = "____________________________"

    ws5["A1"].font = Font(bold=True, color="1F4E78", size=13)
    ws5.column_dimensions["A"].width = 70

    # =============================== #
    # ðŸ’¾ XUáº¤T FILE EXCEL
    # =============================== #
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    filename = f"ThongKe_CVDApp_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"

    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ==========================================
# ðŸŒ¿ Trang Kiáº¿n thá»©c Y há»c (cho bá»‡nh nhÃ¢n)
# ==========================================
@app.route('/tips')
def tips():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Chá»‰ cho phÃ©p bá»‡nh nhÃ¢n xem
    if session.get('role') != 'patient':
        flash("Chá»‰ bá»‡nh nhÃ¢n má»›i Ä‘Æ°á»£c truy cáº­p trang nÃ y.", "warning")
        return redirect(url_for('home'))
    
    return render_template('tips.html')

# ============================================
# ðŸ¤– API CHAT AI (AJAX) â€” NÃ¢ng cáº¥p chuyÃªn nghiá»‡p
# ============================================
@app.route('/chat_ai_api', methods=['POST'])
def chat_ai_api():
    if 'user' not in session or session.get('role') != 'patient':
        return jsonify({'reply': 'âš ï¸ Báº¡n chÆ°a Ä‘Äƒng nháº­p hoáº·c khÃ´ng cÃ³ quyá»n truy cáº­p.'}), 403

    import google.generativeai as genai
    from datetime import datetime
    from flask import jsonify

    data = request.get_json()
    msg = data.get('message', '').strip()
    if not msg:
        return jsonify({'reply': 'ðŸ“ Vui lÃ²ng nháº­p cÃ¢u há»i cá»§a báº¡n.'})

    try:
        # --- Cáº¥u hÃ¬nh model (Ä‘Ã£ cáº¥u hÃ¬nh sáºµn API KEY á»Ÿ Ä‘áº§u file) ---
        model = genai.GenerativeModel(MODEL_NAME)

        # --- Prompt chuyÃªn nghiá»‡p ---
        prompt = f"""
        Báº¡n lÃ  **Trá»£ lÃ½ y táº¿ áº£o CVD-AI**, chuyÃªn tÆ° váº¥n vá» **bá»‡nh tim máº¡ch, huyáº¿t Ã¡p, tiá»ƒu Ä‘Æ°á»ng, lá»‘i sá»‘ng lÃ nh máº¡nh**.
        - Tráº£ lá»i ngáº¯n gá»n, rÃµ rÃ ng, dá»… hiá»ƒu, dÃ¹ng tiáº¿ng Viá»‡t tá»± nhiÃªn.
        - Giá»¯ giá»ng vÄƒn **thÃ¢n thiá»‡n, chuyÃªn nghiá»‡p**, trÃ¡nh dÃ¹ng tá»« ngá»¯ phá»©c táº¡p y há»c.
        - Náº¿u cÃ¢u há»i ngoÃ i chá»§ Ä‘á» sá»©c khá»e, hÃ£y nÃ³i nháº¹ nhÃ ng: 
          â€œXin lá»—i, tÃ´i chá»‰ cÃ³ thá»ƒ tÆ° váº¥n vá» sá»©c khá»e vÃ  tim máº¡ch thÃ´i nhÃ© â€.
        - CÃ³ thá»ƒ chia cÃ¢u tráº£ lá»i thÃ nh 2-3 Ä‘oáº¡n rÃµ rÃ ng.
        - Náº¿u liÃªn quan Ä‘áº¿n thÃ³i quen, gá»£i Ã½ **thá»±c hÃ nh cá»¥ thá»ƒ** (vÃ­ dá»¥: â€œNÃªn táº­p thá»ƒ dá»¥c 30 phÃºt má»—i ngÃ yâ€).
        - KhÃ´ng dÃ¹ng markdown náº·ng (chá»‰ gáº¡ch Ä‘áº§u dÃ²ng, emoji nháº¹).
        
        ðŸ“© CÃ¢u há»i tá»« bá»‡nh nhÃ¢n: 
        {msg}
        """

        # --- Gá»i Gemini API ---
        response = model.generate_content(prompt)

        answer = response.text.strip() if response and response.text else (
            "ðŸ¤” Xin lá»—i, tÃ´i chÆ°a hiá»ƒu rÃµ cÃ¢u há»i cá»§a báº¡n. Báº¡n cÃ³ thá»ƒ diá»…n Ä‘áº¡t láº¡i Ä‘Æ°á»£c khÃ´ng?"
        )

        # --- LÃ m Ä‘áº¹p pháº£n há»“i: xá»­ lÃ½ format nháº¹ ---
        formatted_answer = (
            answer.replace("**", "")  # bá» markdown Ä‘áº­m
                  .replace("* ", "â€¢ ")  # thay bullet
                  .replace("#", "")
        )

        # --- LÆ°u vÃ o cÆ¡ sá»Ÿ dá»¯ liá»‡u ---
        user_id = session.get('user_id')
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO TinNhanAI (BenhNhanID, NoiDung, PhanHoi, ThoiGian)
            VALUES (?, ?, ?, ?)
        """, (user_id, msg, formatted_answer, datetime.now()))
        conn.commit()
        conn.close()

        # --- Tráº£ vá» káº¿t quáº£ cho giao diá»‡n ---
        return jsonify({'reply': formatted_answer})

    except Exception as e:
        print("âš ï¸ Lá»—i Gemini AI:", e)
        return jsonify({
            'reply': 'ðŸš« Há»‡ thá»‘ng AI Ä‘ang báº­n hoáº·c káº¿t ná»‘i khÃ´ng á»•n Ä‘á»‹nh. Vui lÃ²ng thá»­ láº¡i sau Ã­t phÃºt.'
        })
# ==========================================
# ðŸ“œ API láº¥y lá»‹ch sá»­ chat AI cá»§a ngÆ°á»i dÃ¹ng hiá»‡n táº¡i
# ==========================================
@app.route('/chat_ai_history', methods=['GET'])
def chat_ai_history():
    if 'user' not in session or session.get('role') != 'patient':
        return jsonify({'messages': []}), 403

    user_id = session['user_id']
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT NoiDung, PhanHoi, FORMAT(ThoiGian, 'HH:mm dd/MM') AS ThoiGian
        FROM TinNhanAI
        WHERE BenhNhanID = ?
        ORDER BY ThoiGian
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()

    messages = [
        {'user': r.NoiDung, 'ai': r.PhanHoi, 'time': r.ThoiGian}
        for r in rows
    ]
    return jsonify({'messages': messages})

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    app.run(debug=True)

