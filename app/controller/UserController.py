from app import app, db
from app.model.user import User
from app.helpers.commons import generate_confirmation_token, send_mail


from flask import request, flash, redirect, url_for, render_template, session
from flask_login import current_user
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os
import ast

now = datetime.now()
expired = now + timedelta(hours=1)

def register():
    try:
        nama = request.form.get('nama')
        jkel = request.form.get('jkel')
        telepon = request.form.get('telepon')
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(username=email).first()
        if user:
            flash("Gagal! Email sudah digunakan.", "error")
        else:
            subject = "Aktivasi Akun"
            content = render_template(
                'mail/activation.html', 
                confirm_url = url_for(
                                    'auth.activate', 
                                    token = generate_confirmation_token(email),
                                    _external=True),
                nama=nama,
                expire=expired.strftime("tanggal %d %B %Y pukul %H:%M:%S"))
            
            send_mail(email, subject, content)

            user = User(nama=nama, jkel=jkel, telepon=telepon, username=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()

            flash("Tautan aktivasi telah dikirim ke inbox email Anda.", "success")
    
    except Exception as e:
        print(e)
        flash("Akun gagal dibuat", "error")
    finally:
        return redirect(url_for('auth.login'))
    
def createUser():
    nama = request.form.get('nama')
    jkel = request.form.get('jkel')
    telepon = request.form.get('telepon')
    email = request.form.get('email')
    password = request.form.get('password')
    is_admin = request.form.get('is_admin')

    user = User.query.filter_by(username=email).first()

    if user:
        flash('User dengan email {} sudah terdaftar.'.format(email), 'error')
    else:
        try:
            new_user = User(nama=nama, jkel=jkel, telepon=telepon, username=email)
            new_user.set_password(password)
            if is_admin == 'on':
                new_user.is_admin = True
                role = 'Admin'
            else:
                role = 'Pengguna'
            db.session.add(new_user)
            db.session.commit()

            subject = "Undangan pengguna baru"
            content = render_template(
                'mail/invitation.html', 
                confirm_url = url_for(
                                    'auth.activate', 
                                    token = generate_confirmation_token(email),
                                    _external=True),
                nama=nama,
                email=email,
                password=password,
                pengundang=current_user.nama,
                role=role,
                expire=expired.strftime("tanggal %d %B %Y pukul %H:%M:%S"))
            
            send_mail(email, subject, content)
            flash('Berhasil menambahkan. Email aktivasi sudah dikirim ke pengguna', 'success')

        except Exception as e:
            print(e)
            flash('Gagal menambah pengguna!', 'error')
        finally:
            return redirect(url_for('user.manage_users'))

def profile():
    user = User.query.filter_by(username=current_user.username).first()
    return render_template('user/profile.html', user=user)

def update_profile():
    try:
        nama = request.form.get('nama')
        jkel = request.form.get('jkel')
        email = request.form.get('email')
        telepon = request.form.get('telepon')
        pekerjan = request.form.get('pekerjaan')
        alamat = request.form.get('alamat')

        user = User.query.filter_by(id=current_user.id).first()
        user.nama = nama
        user.jkel = jkel
        user.username = email
        user.telepon = telepon
        user.pekerjaan = pekerjan
        user.alamat = alamat

        db.session.commit()
        flash('Berhasil merubah profil!', 'success')
    except Exception as e:
        print(e)
        flash('Gagal merubah profil!', 'error')
    finally:
        return redirect(url_for('user.profile'))

def upload_img():
    try:
        user = User.query.filter_by(id=current_user.id).first()
        uploaded_img = request.files['uploaded_img']
        img_filename = secure_filename(uploaded_img.filename)
        extension = img_filename.split('.')[1]
        file_name = "profile-img/user-"+str(user.id)+"."+extension
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        if os.path.exists(file_path):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
            
        uploaded_img.save(file_path)
    
        user.foto = file_name
        db.session.commit()
        flash('Foto profil berhasil diupload.','success')
    except Exception as e:
        print(e)
        flash('Gagal upload foto profil!','error')
    finally:
        return redirect(url_for('user.profile'))

def update_password():
    try:
        userid = current_user.id
        new_password = request.form.get('new_password')
        
        user = User.query.filter_by(id=userid).first()
        user.set_password(new_password)
        db.session.commit()
            
        flash('Password berhasil diubah.', 'success')                                                                               
    except:
        flash("Gagal mengubah password :(", "error")
    finally:
        return redirect(url_for('user.profile'))
    
def manage_users():
    users = User.query.all()
    return render_template('user/users.html', users=users)

def confirm_user(id, is_confirmed):
    try:
        user = User.query.filter_by(id=id).first()
        user.is_confirmed = ast.literal_eval(is_confirmed)
        user.confirmed_on = datetime.now()
        db.session.commit()
        if user.is_confirmed:
            flash(f"Berhasil mengaktifkan user '{user.nama}'.", "success")
        else:
            flash(f"Berhasil menonaktifkan user '{user.nama}'.", "success")
    except Exception as e:
        print(e)
        flash(f"Gagal merubah status user {e}.", "error")
    finally:
        return redirect(url_for('user.manage_users'))
    
def set_role(id, is_admin):
    try:
        user = User.query.filter_by(id=id).first()
        user.is_admin = ast.literal_eval(is_admin)
        db.session.commit()
        if user.is_admin:
            flash(f"Berhasil merubah user '{user.nama}' jadi admin.", "success")
        else:
            flash(f"Berhasil merubah admin '{user.nama}' jadi user.", "success")
    except Exception as e:
        print(e)
        flash(f"Gagal merubah peran user {e}.", "error")
    finally:
        return redirect(url_for('user.manage_users'))
    
def delete_user(id):
    try:
        user = User.query.filter_by(id=id).first()
        flash(f"Berhasil menghapus {user.nama}.", "success")
        db.session.delete(user)
        db.session.commit()
    except Exception as e:
        print(e)
        flash(f"Gagal menghapus user {e}.", "error")
    finally:
        return redirect(url_for('user.manage_users'))

        