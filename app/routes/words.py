from flask import Blueprint, request, render_template, redirect, url_for, session
from app.controller import WordsController
from flask_login import login_required
from app.restriction import admin_required

words = Blueprint('words', __name__)

@words.route('/informal', methods=['GET','POST'])
def informal():
    if request.method == 'POST':
        return WordsController.add_formal()
    else:
        return WordsController.formal()
    
@words.route('/informal/update/<id>', methods=['POST'])
@login_required
@admin_required
def edit_informal(id):
    return WordsController.edit_formal(id)
    
@words.route('/informal/delete/<id>')
@login_required
@admin_required
def delete_informal(id):
    return WordsController.delete_formal(id)

@words.route('/negasi', methods=['GET','POST'])
def negation():
    if request.method == 'POST':
        return WordsController.add_negation()
    else:
        return WordsController.negation()
    

@words.route('/negasi/update/<id>', methods=['POST'])
@login_required
@admin_required
def edit_negation(id):
    return WordsController.edit_negation(id)


@words.route('/negasi/delete/<id>')
@login_required
@admin_required
def delete_negation(id):
    return WordsController.delete_negation(id)


@words.route('/root', methods=['GET','POST'])
def root():
    if request.method == 'POST':
        return WordsController.add_root()
    else:
        return WordsController.root()
    

@words.route('/root/update/<id>', methods=['POST'])
@login_required
@admin_required
def edit_root(id):
    return WordsController.edit_root(id)


@words.route('/root/delete/<id>')
@login_required
@admin_required
def delete_root(id):
    return WordsController.delete_root(id)

@words.route('/compound', methods=['GET','POST'])
def compound():
    if request.method == 'POST':
        return WordsController.add_compound()
    else:
        return WordsController.compound()
    

@words.route('/compound/update/<id>', methods=['POST'])
@login_required
@admin_required
def edit_compound(id):
    return WordsController.edit_compound(id)


@words.route('/compound/delete/<id>')
@login_required
@admin_required
def delete_compound(id):
    return WordsController.delete_compound(id)


@words.route('/stopword', methods=['GET','POST'])
def stopword():
    if request.method == 'POST':
        return WordsController.add_stopword()
    else:
        return WordsController.stopword()
    

@words.route('/stopword/update/<id>', methods=['POST'])
@login_required
@admin_required
def edit_stopword(id):
    return WordsController.edit_stopword(id)


@words.route('/stopword/delete/<id>')
@login_required
@admin_required
def delete_stopword(id):
    return WordsController.delete_stopword(id)


@words.route('/positive', methods=['GET','POST'])
def positive():
    if request.method == 'POST':
        return WordsController.add_positive()
    else:
        return WordsController.positive()
    

@words.route('/positive/update/<id>', methods=['POST'])
@login_required
@admin_required
def edit_positive(id):
    return WordsController.edit_positive(id)


@words.route('/positive/delete/<id>')
@login_required
@admin_required
def delete_positive(id):
    return WordsController.delete_positive(id)


@words.route('/negative', methods=['GET','POST'])
def negative():
    if request.method == 'POST':
        return WordsController.add_negative()
    else:
        return WordsController.negative()
    

@words.route('/negative/update/<id>', methods=['POST'])
@login_required
@admin_required
def edit_negative(id):
    return WordsController.edit_negative(id)


@words.route('/negative/delete/<id>')
@login_required
@admin_required
def delete_negative(id):
    return WordsController.delete_negative(id)


@words.route('/corpus', methods=['GET','POST'])
def corpus():
    if request.method == 'POST':
        return WordsController.add_corpus()
    else:
        return WordsController.corpus()
    

@words.route('/corpus/update/<id>', methods=['POST'])
@login_required
@admin_required
def edit_corpus(id):
    return WordsController.edit_corpus(id)


@words.route('/corpus/delete/<id>')
@login_required
@admin_required
def delete_corpus(id):
    return WordsController.delete_corpus(id)