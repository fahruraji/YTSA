from flask import Blueprint, request, render_template, redirect, url_for, session, Response
import os

from app import app, db
from app.controller import HomeController, MainController
from flask_login import login_required
from app.restriction import admin_required

from app.model import *
from app.helpers.commons import get_columns

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def home():
     if request.method == 'POST':
         return HomeController.classify()
     else:
          return HomeController.index()

@main.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    if request.method == 'POST':
        return MainController.search_youtube()
    else:
        return MainController.search()
    
@main.route('/history', methods=['GET', 'POST'])
@login_required
def history():
    if request.method == 'POST':
        return MainController.get_result()
    else:
        return MainController.history()
    
@main.route('/history/delete/<id>')
@login_required
def delete_history(id):
    return MainController.delete_history(id)

@main.route('/scraping/', methods=['POST'])
@main.route('/scraping/<id>', methods=['GET'])
@login_required
def scraping(id=None):
    return MainController.scraping(id)
    
@main.route('/preprocessing/')
@login_required
def preprocessing():
    return MainController.preprocessing()

@main.route('/processing/')
@login_required
def processing():
    return MainController.processing()

@main.route('/generate_image/')
@login_required
def generate_image():
    return MainController.generate_image()

@main.route('/edit_normalisasi/', methods=['POST'])
@login_required
def edit_normalisasi():
    return MainController.edit_normalisasi()

@main.route('/add_feedback', methods=['POST'])
@login_required
def add_feedback():
    return MainController.add_feedback()

@main.route('/result')
@login_required
def result():
    return MainController.result()

@main.route('/train', methods=['GET', 'POST'])
@login_required
def train_model():
    return MainController.train_model()

@main.route('/model/<string:mode>', methods=['GET', 'POST'])
@login_required
def train_result(mode):
    return MainController.train_result(mode)

from sqlalchemy import func
from sqlalchemy.sql import label

@main.route('/tes', methods=['GET', 'POST'])
def tes():
    return MainController.tes()
    # data = dict(db.session.query(
    #         label('Intent', Processed.intent),  func.count(Processed.intent)
    # ).group_by(Processed.intent).all())
    # # data = dict(data)
    # return render_template('tes.html', data=data)
