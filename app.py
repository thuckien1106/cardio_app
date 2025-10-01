from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "secret-key-demo"

# Thư mục upload
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# =====================================
# Đăng nhập
# =====================================
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('username')
        pw = request.form.get('password')

        # Demo tài khoản
        if user == 'doctor' and pw == '123':
            session['user'] = user
            session['role'] = 'doctor'
            return redirect(url_for('home'))           # ➡️ Vào trang Home
        elif user == 'patient' and pw == '123':
            session['user'] = user
            session['role'] = 'patient'
            return redirect(url_for('home'))           # ➡️ Vào trang Home
        else:
            return render_template('login.html', error="Sai tài khoản hoặc mật khẩu")

    return render_template('login.html')


# =====================================
# Trang Home (sau khi đăng nhập)
# =====================================
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')


# =====================================
# Trang chẩn đoán (chỉ cho bác sĩ)
# =====================================
@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    if 'user' not in session:
        return redirect(url_for('login'))

    # Ngăn bệnh nhân vào trang chẩn đoán
    if session.get('role') == 'patient':
        flash("Bạn không có quyền truy cập trang này.")
        return redirect(url_for('history'))

    result = None
    file_result = None

    # ----- Xử lý nhập form -----
    if request.method == 'POST' and 'predict_form' in request.form:
        age = int(request.form.get('age'))
        gender = request.form.get('gender')
        weight = float(request.form.get('weight'))
        height = float(request.form.get('height'))
        systolic = float(request.form.get('systolic'))
        diastolic = float(request.form.get('diastolic'))
        chol = request.form.get('cholesterol')     # mức độ
        glucose = request.form.get('glucose')      # mức độ
        smoking = request.form.get('smoking')
        alcohol = request.form.get('alcohol')
        exercise = request.form.get('exercise')

        # Tính BMI
        bmi = round(weight / ((height / 100) ** 2), 2)

        # ----- Logic dự đoán demo -----
        risk_score = 0

        # Huyết áp
        if systolic > 140 or diastolic > 90:
            risk_score += 1

        # Cholesterol (3 mức)
        if chol == 'above_normal':
            risk_score += 1
        elif chol == 'high':
            risk_score += 2

        # Đường huyết (3 mức)
        if glucose == 'above_normal':
            risk_score += 1
        elif glucose == 'high':
            risk_score += 2

        # BMI
        if bmi > 30:
            risk_score += 1

        # Thói quen
        if smoking == 'yes':
            risk_score += 1
        if alcohol == 'yes':
            risk_score += 1

        result = f"Nguy cơ cao (BMI: {bmi})" if risk_score >= 3 else f"Nguy cơ thấp (BMI: {bmi})"

    # ----- Xử lý upload file -----
    if request.method == 'POST' and 'data_file' in request.files:
        file = request.files['data_file']
        if file.filename != '':
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            # Đọc dữ liệu
            if filename.endswith('.csv'):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)

            # Gán nhãn demo
            df['Prediction'] = ['Cao' if i % 2 == 0 else 'Thấp' for i in range(len(df))]
            file_result = df.to_html(classes='table table-striped table-bordered', index=False)

    return render_template('diagnose.html', result=result, file_result=file_result)


# =====================================
# Trang lịch sử chẩn đoán
# =====================================
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))
    # Ở đây có thể load lịch sử từ DB theo session['user']
    return render_template('history.html')


# =====================================
# Trang bệnh án (chỉ bác sĩ)
# =====================================
@app.route('/records')
def records():
    if 'user' not in session:
        return redirect(url_for('login'))

    if session.get('role') == 'patient':
        flash("Bạn không có quyền truy cập trang này.")
        return redirect(url_for('history'))

    return render_template('records.html')


# =====================================
# Trang hồ sơ
# =====================================
@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))

    # Demo dữ liệu
    if session.get('role') == 'doctor':
        user_info = {
            'name': 'Bác sĩ Demo',
            'email': 'doctor@cvdapp.com',
            'role': 'Bác sĩ',
            'phone': '0909 123 456'
        }
    else:
        user_info = {
            'name': 'Bệnh nhân Demo',
            'email': 'patient@cvdapp.com',
            'role': 'Bệnh nhân',
            'phone': '0912 345 678'
        }

    return render_template('profile.html', user_info=user_info)


# =====================================
# Đăng xuất
# =====================================
@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('role', None)
    flash("Bạn đã đăng xuất!")
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
