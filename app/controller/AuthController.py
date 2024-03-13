from app import app, db
from app.model import User
from app.helpers.commons import confirm_token, send_mail

from flask import request, flash, redirect, url_for, render_template, session
from flask_login import login_user, logout_user
from datetime import datetime
import pyotp
from datetime import datetime, timedelta

now = datetime.now()
expired = now + timedelta(hours=1)

def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    user = User.query.filter_by(username=username).first()
    if not user:
        flash("Alamat email belum terdaftar", "error")
        return redirect(url_for('auth.login'))
    if not user.check_password(password):
        flash("Kombinasi password salah", "error")
        return redirect(url_for('auth.login'))

    if user.is_confirmed:
        login_user(user)
        if user.alamat == None or user.pekerjaan == None:
            flash("Silakan lengkapi info pekerjaan dan alamat Anda.", "warning")
            return redirect(url_for('user.profile'))
        else:
            return redirect(url_for('main.home'))
    else:
        flash("Akun Anda belum aktif. Silakan klik tautan dalam email aktivasi yang kami kirimkan!.", "error")
        return redirect(request.referrer)

def logout():
    logout_user()
    return redirect(url_for("auth.login"))

def activate(token):
    email, new_token = confirm_token(token)
    confirmed_on = datetime.now()

    user = User.query.filter_by(username=email).first()

    if new_token:
        flash(f'Tautan sudah kadaluwarsa. Kami telah mengirimkan ulang tautan aktivasi yang baru', 'error')
        subject = "Aktivasi Akun"
        content = render_template(
            'mail/activation.html',
            confirm_url = url_for(
                                'auth.activate', 
                                token = new_token,
                                _external=True),
            nama=user.nama,
            expire=expired.strftime("tanggal %d %B %Y pukul %H:%M:%S"))
        
        send_mail(email, subject, content)
    else:
        if not user.is_confirmed:
            try:
                user.is_confirmed = True
                user.confirmed_on = confirmed_on
                db.session.commit()

                flash("Akun Anda berhasil diaktivasi. Silakan login.", "success")
            except Exception as e:
                print(e)
                flash("Gagal mengaktifkan akun.", "error")
        else:
            flash('Akun sudah aktif. Tidak perlu aktivasi lagi', 'warning')
    
    return redirect(url_for('auth.login'))

def send_otp():
    username = request.form.get('email')
    telepon = request.form.get('phone')

    user = User.query.filter_by(username=username).first()

    if user and user.is_confirmed:
        if user.telepon == telepon:
            otp = pyotp.random_base32()
            subject = "Kode Verifikasi Reset Password"
            content = render_template(
                'mail/verification.html', 
                otp = otp,
                nama = user.nama)
            send_mail(username, subject, content)

            session['current_otp'] = otp
            session['email'] = username

            flash('Kode verifikasi telah kami kirimkan ke inbox email Anda.', 'success')
            return redirect(url_for('auth.reset_passwd'))
        else:
            flash("No. Telp tidak dikenali. Cek kembali No. Telp yang Anda masukkan!", "error")
            return redirect(url_for('auth.login'))
    else:
        flash("Akun belum diaktivasi atau belum dibuat!", "error")
        return redirect(url_for('auth.login'))
    

def reset_passwd():
    username = request.form.get('email')
    new_password = request.form.get('new_password')
    otp = request.form.get('otp')

    user = User.query.filter_by(username=username).first()
    if session.get('current_otp') == otp:
        user.set_password(new_password)
        db.session.commit()

        del session['current_otp']
        del session['email']

        flash("Password telah diubah", "success")
        return redirect(url_for('auth.login'))
    else:
        flash("Kode verifikasi salah. Cek email untuk melihat kode verifikasi", "error")
        return redirect(url_for('auth.reset_passwd'))
    