from flask import Blueprint, request, render_template, redirect, url_for, session
from app.controller import AuthController, UserController

auth = Blueprint('auth', __name__)

@auth.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        return UserController.register()
    
@auth.route('/reset_passwd', methods=['GET', 'POST'])
def reset_passwd():
    if request.method == 'POST':
        return AuthController.reset_passwd()
    else:
        if 'current_otp' in session:
            return render_template('auth/reset_passwd.html')
        else:
            return redirect(url_for('auth.login'))
    
@auth.route('/activate/<token>')
def activate(token):
     return AuthController.activate(token)

@auth.route('/send_otp', methods=['POST'])
def send_otp():
     return AuthController.send_otp()

@auth.route('/login', methods=['GET', 'POST'])
def login():
     if request.method == 'POST':
        return AuthController.login()
     else:
        return render_template('auth/login.html')
     
@auth.route('/logout')
def logout():
     return AuthController.logout()