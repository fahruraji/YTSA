from app import app
from app.helpers.classifying import IndoBERTClassify, SVMClassify
from app.helpers.preprocessing import preprocessing
from flask import render_template, redirect, url_for, request, session
import os

def index():
    result = session.pop('result', None)
    models_path = ['app/ml/svm/sentiment/model.joblib', 'app/ml/svm/intent/model.joblib', 'app/ml/indobert/sentiment/model.safetensors', 'app/ml/indobert/intent/model.safetensors']
    for model_path in models_path:
        if os.path.exists(model_path):
            model_exist = True
        else:
            model_exist = None
    return render_template('home/index.html', result=result, model_exist=model_exist)

def classify():
    sentence = request.form['sentence']
    svm_sentiment = SVMClassify('sentiment')
    svm_intent = SVMClassify('intent')
    indobert_sentiment = IndoBERTClassify('sentiment')
    indobert_intent = IndoBERTClassify('intent')
    text_prep = preprocessing(sentence)
    result = {
                'sentence' : sentence,
                'svm_sentiment': svm_sentiment.predict(text_prep)['label'],
                'svm_intent': svm_intent.predict(text_prep)['label'],
                'indobert_sentiment': indobert_sentiment.predict(sentence)['label'],
                'indobert_intent': indobert_intent.predict(sentence)['label'],
            }
    session['result'] = result
    return redirect(url_for('main.home'))