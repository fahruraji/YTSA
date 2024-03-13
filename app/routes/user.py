from flask import Blueprint, request, render_template, redirect, url_for, session
from app.controller import HomeController, UserController
from flask_login import login_required
from app.restriction import admin_required

user = Blueprint('user', __name__)

@user.route('/profil', methods=['GET', 'POST'])
@login_required
def profile():
     if request.method == 'POST':
          return UserController.update_profile()
     else:
          return UserController.profile()

@user.route('/profil/upload-img',  methods=["POST"])
@login_required
def upload_img():
     if request.method == 'POST':
          return UserController.upload_img()

@user.route('/profil/update-password',  methods=["POST"])
@login_required
def update_password():
     if request.method == 'POST':
          return UserController.update_password()

@user.route('/users', methods=['GET', 'POST'])
@login_required
@admin_required
def manage_users():
     if request.method == 'POST':
          return UserController.createUser()
     else:
          return UserController.manage_users()

@user.route('/user/active/<id>/<is_confirmed>')
@login_required
@admin_required
def set_state(id, is_confirmed):
     return UserController.confirm_user(id, is_confirmed)

@user.route('/user/admin/<id>/<is_admin>')
@login_required
@admin_required
def set_role(id, is_admin):
     return UserController.set_role(id, is_admin)

@user.route('/user/delete/<id>')
@login_required
@admin_required
def delete_user(id):
     return UserController.delete_user(id)