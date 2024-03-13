from flask import request, render_template, redirect, url_for, session, flash, jsonify
from flask_paginate import Pagination, get_page_args
from flask_login import current_user
from sqlalchemy import func, or_, and_
from sqlalchemy.sql import label
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import numpy as np
import joblib
import csv
import os

from app import app, db
from app.helpers.classifying import IndoBERTFineTuner, IndoBERTClassify, SVMModel, SVMClassify
from app.helpers.scraping import get_ids, get_info, scrape_comments, is_url, is_youtube_url
from app.helpers.preprocessing import casefolding, tokenizing, filtering, normalizing, stemming, preprocessing_1, preprocessing_2
from app.helpers.visualizing import dist_freq, data_to_json, generate_wordcloud
from app.helpers.commons import paginating
from app.model import Youtube, Analysis, Comments, Preprocessed, Processed, Corpus

indobert_sentiment = IndoBERTClassify('sentiment')
indobert_intent = IndoBERTClassify('intent')
svm_sentiment = SVMClassify('sentiment')
svm_intent = SVMClassify('intent')

def index():
    result = session.pop('result', None)
    return render_template('home/index.html', result=result)

def search():
    keyword = request.args.get('q')
    if keyword:
        keyword = keyword.split('v=')[1] if is_youtube_url(keyword) else keyword
        existing_entry = Youtube.query.filter(or_(Youtube.title.contains(keyword), Youtube.id==keyword)).all()        
        if not existing_entry and len(existing_entry) < 20:
            results = get_info(keyword) if get_info(keyword) else get_ids(keyword)
            for result in results:
                try:
                    info = Youtube(**result)
                    db.session.add(info)
                    db.session.commit()
                except:
                    pass

        if get_info(keyword):
            analyzed = Analysis.query.filter_by(youtube_id=keyword, user_id=current_user.id).first()
            if not analyzed:
                return redirect(url_for('main.scraping', id=keyword))
            else:
                flash('Anda sudah pernah melakukan analisis terhadap video tersebut. Hapus terlebih dahulu dari riwayat analisis, jika Anda ingin menganalisis ulang video.', 'error')
                return redirect(url_for('main.history'))
        else:
            result, pagination = paginating(Youtube, order_by=Youtube.views.desc(), filtering=Youtube.title.contains(keyword), filtering2=Youtube.description.contains(keyword), rows_per_page=12)
            return render_template('main/search.html', result=result, pagination=pagination, keyword=keyword, total=result.total)
        
    else:
        return render_template('main/search.html')    

def scraping(id=None):
    if id:
        vid_id = id
    else:
        vid_id = request.form.get('id')
    analyzed = Analysis.query.filter_by(youtube_id=vid_id, user_id=current_user.id).first()
    if analyzed:
        flash("Anda sudah pernah melakukan analisis terhadap video tersebut. Hapus terlebih dahulu dari riwayat analisis, jika Anda ingin menganalisis ulang video.", 'error')
        return redirect(request.referrer)
    else:
        try:
            results = scrape_comments(vid_id)
            analysis = Analysis(youtube_id=vid_id, user_id=current_user.id)
            db.session.add(analysis)
            db.session.commit()

            for result in results:
                comment = Comments(
                    analysis_id=analysis.id,
                    title=result['title'],
                    name=result['name'],
                    comment=result['comment'],
                    published_at=result['published_at'],
                    likes=result['likes'],
                    replies=result['replies'])
                db.session.add(comment)
                db.session.commit()

            session['analysis_id'] = analysis.id      

        except Exception as e:
            print(e)
            flash('Gagal mengambil komentar.', 'error')
        finally:
            return redirect(url_for('main.preprocessing'))

def preprocessing():
    try:
        comments = Comments.query.filter_by(analysis_id=session['analysis_id']).all()
        for row in comments:
            comment_id = row.id
            casefolded = casefolding(row.comment)
            tokenized = tokenizing(casefolded)
            normalized = normalizing(tokenized)
            stemmed = stemming(normalized)
            filtered = filtering(stemmed)

            data = Preprocessed(
                comment_id=comment_id,
                casefolded=casefolded,
                tokenized=', '.join(map(str, tokenized)),
                # normalized=' '.join(map(str, normalized)),
                # filtered=' '.join(map(str, filtered)),
                normalized=normalized,
                filtered=filtered,
                stemmed=stemmed
            )
            db.session.add(data)
            db.session.commit()
    except Exception as e:
        print(e)
        flash(f'Gagal melakukan preprocessing: {e}', 'error')
    finally:
        return redirect(url_for('main.processing'))
    
def processing():    
    try:
        comments = Preprocessed.query.join(Comments).filter(Comments.analysis_id==session['analysis_id'], Preprocessed.casefolded.isnot(None), Preprocessed.casefolded != '').all()
        # vectors = []
        for comment in comments:
            vectors = svm_sentiment.predict(comment.stemmed)['vector']
            encodes = indobert_sentiment.predict(comment.normalized)['vector']

            data = Processed(
                comment_id = comment.id,
                vectors = ', '.join(map(str, vectors)),
                svm = svm_sentiment.predict(comment.stemmed)['label'],
                encodes = ', '.join(map(str, encodes.tolist())),
                indobert = indobert_sentiment.predict(comment.normalized)['label'],
                intent = indobert_intent.predict(comment.normalized)['label'],
            )
            db.session.add(data)
            db.session.commit()
            # vectors.append(comment)

        return redirect(url_for('main.generate_image'))
        # return render_template("tes.html", data=comment, data2=vectors)
        
    except Exception as e:
        print(e)
        flash(f'Gagal melakukan klasifikasi {e}.', 'error')
        
    
def generate_image():
    try:
        id = session['analysis_id']

        mask_image = np.array(Image.open(os.path.join('app', 'static', 'img', 'chat.png')))

        models = ['svm', 'indobert']
        sentiments = ['positif', 'negatif']

        for model in models:
            for sentiment in sentiments:
                data = ' '.join(item.comment.filtered for item in Processed.query.join(Preprocessed).join(Comments).filter(Comments.analysis_id==id, getattr(Processed,model)==sentiment).all()).split()
                # data = Processed.query.join(Preprocessed).join(Comments).filter(Comments.analysis_id==id, getattr(Processed,model)==sentiment).all().split()
                if data:
                    generate_wordcloud(data, sentiment, model+'_'+str(id), mask_image)

    except Exception as e:
        print(e)
        flash('Gagal membuat gambar.', 'error')
    finally:
        return redirect(url_for('main.result'))

def edit_normalisasi():
    edited_data = request.json
    id = edited_data.get('id')
    new_value = edited_data.get('value')
    old_value = Preprocessed.query.filter(Preprocessed.id==id).first().normalized

    if old_value != new_value:
        try:          
            casefolded = casefolding(new_value)
            tokenized = tokenizing(casefolded)
            normalized = normalizing(tokenized)
            stemmed = stemming(normalized)
            filtered = filtering(stemmed)

            preprocessing = Preprocessed.query.filter(Preprocessed.id==id).first()
            preprocessing.normalized = normalized
            preprocessing.filtered = filtered
            preprocessing.stemmed = stemmed
            db.session.commit()

            session['analysis_id'] = preprocessing.comment.analysis_id
            vectors = svm_sentiment.predict(stemmed)['vector']
            encodes = indobert_sentiment.predict(normalized)['vector']

            processing = Processed.query.filter_by(comment_id=id).first()
            processing.vectors = ', '.join(map(str, vectors))
            processing.svm = svm_sentiment.predict(stemmed)['label']
            processing.encodes = ', '.join(map(str, encodes.tolist()))
            processing.indobert = indobert_sentiment.predict(normalized)['label']
            processing.intent = indobert_intent.predict(normalized)['label']
            db.session.commit()

            return jsonify({'status': 'success', 'message': 'Sukses mengedit data.'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    else:
        pass

def add_feedback():
    id = request.form.get('editId')
    text = request.form.get('edit1')
    sentiment = request.form.get('edit2')
    intent = request.form.get('edit3')

    prep_1 = preprocessing_1(text)
    prep_2 = preprocessing_2(prep_1)

    try:
        data_exist = Corpus.query.filter(Corpus.id==id).first()
        if not data_exist:
            data = Corpus(
                text = text,
                prep_1 = prep_1,
                prep_2 = prep_2,
                sentiment = sentiment,
                intent = intent,
                kontributor_id = current_user.id,
            )
            db.session.add(data)
            db.session.commit()

            processed = Processed.query.filter(Processed.id==id).first()
            processed.indobert = sentiment
            processed.intent = intent
            processed.feedback = True
            db.session.commit()

            generate_image()
            flash('Sukses memberikan feedback.', 'success')
            return redirect(request.referrer)
    except Exception as e:
        flash('Gagal memberikan feedback: {e}.', 'error')     

def result():
    def get_max(data):
        max_value = max(data.values(), default=0)
        max_items = [key for key, value in data.items() if value == max_value]
        result = ' dan '.join(max_items)
        return result, len(max_items)
    def summary(data):
        label, length = get_max(data)[0], get_max(data)[1]
        if length > 1:
            return f'memberikan respon yang sama besar antara respon {label}'
        else:
            if label == 'positif':
                return 'memberikan respon positif'
            elif label == 'negatif':
                return 'berpandangan negatif'
            elif label == 'netral':
                return 'bersikap netral'
            
    def get_sentiment_stats(id, sentiment_column):
        intent_labels = [value[0] for value in db.session.query(Processed.intent).join(Preprocessed).join(Comments).filter(Comments.analysis_id==id).distinct().all()]

        # Inisialisasi data statistik sentimen
        sentiment_stats = [['Intent', 'positif', 'negatif', 'netral']]

        for intent_label in intent_labels:
            query_result = db.session.query(label('Sentiment', getattr(Processed, sentiment_column)), func.count(getattr(Processed, sentiment_column))) \
                .join(Preprocessed).join(Comments).filter(Comments.analysis_id==id, Processed.intent == intent_label) \
                .group_by(getattr(Processed, sentiment_column)).all()

            # Inisialisasi counter untuk setiap sentimen
            sentiment_count = {'positif': 0, 'negatif': 0, 'netral': 0}

            # Mengisi counter berdasarkan hasil query
            for sentiment, count in query_result:
                sentiment_count[sentiment] = count

            # Menambahkan data intent dan statistik sentimen ke hasil akhir
            result_row = [intent_label, sentiment_count['positif'], sentiment_count['negatif'], sentiment_count['netral']]
            sentiment_stats.append(result_row)
        return sentiment_stats
            
    if 'analysis_id' in session:
        id = session['analysis_id']
        title = Analysis.query.filter_by(id=id).first().youtube.title
        comments = Comments.query.filter_by(analysis_id=id).all()
        preprocessed = Preprocessed.query.join(Comments).filter(Comments.analysis_id==id, Preprocessed.casefolded.isnot(None), Preprocessed.casefolded != '').all()
        processed = Processed.query.join(Preprocessed).join(Comments).filter(Comments.analysis_id==id).all()
        # svm = [item.svm for item in Processed.query.join(Preprocessed).join(Comments).filter(Comments.analysis_id==id).all()]
        # indobert = [item.indobert for item in Processed.query.join(Preprocessed).join(Comments).filter(Comments.analysis_id==id).all()]
        # intent = [item.intent for item in Processed.query.join(Preprocessed).join(Comments).filter(Comments.analysis_id==id).all()]
        svm = dict(
            db.session.query(label('Sentiment', Processed.svm), func.count(Processed.svm))
            .join(Preprocessed).join(Comments).filter(Comments.analysis_id==id)
            .group_by(Processed.svm).all())
        indobert = dict(
            db.session.query(label('Sentiment', Processed.indobert), func.count(Processed.indobert))
            .join(Preprocessed).join(Comments).filter(Comments.analysis_id==id)
            .group_by(Processed.indobert).all())
        intent = dict(
            db.session.query(label('Intent', Processed.intent), func.count(Processed.intent))
            .join(Preprocessed).join(Comments).filter(Comments.analysis_id==id)
            .group_by(Processed.intent).all())

        # pos_words = dist_freq([item.comment.stemmed for item in Processed.query.join(Preprocessed).join(Comments).filter(Comments.analysis_id==id, Processed.svm=='Positif').all()], 50)
        words = dist_freq([item.comment.stemmed for item in Processed.query.join(Preprocessed).join(Comments).filter(Comments.analysis_id==id).all()], 50)
        
        pos_words = ' '.join(item.comment.stemmed for item in Processed.query.join(Preprocessed).join(Comments).filter(Comments.analysis_id==id, Processed.svm=='positif').all()).split()
        neg_words = ' '.join(item.comment.stemmed for item in Processed.query.join(Preprocessed).join(Comments).filter(Comments.analysis_id==id, Processed.svm=='negatif').all()).split()  

        return render_template(
            'main/index.html',
            id=str(id), 
            title=title, 
            comments=comments, 
            preprocessed=preprocessed, 
            processed=processed,  
            svm=svm, 
            indobert=indobert,
            intent=intent,
            intent_max=get_max(intent)[0],
            summary = summary(indobert),
            total=len(processed), 
            total_positif=indobert.get('positif', 0), 
            total_negatif=indobert.get('negatif', 0), 
            total_netral=indobert.get('netral', 0),
            dist_words=data_to_json(words, 'Variabel', 'Frekuensi'),
            pos_words=pos_words,
            neg_words=neg_words,
            dist_indobert = get_sentiment_stats(id, 'indobert'),
            )
    else:
        flash('Buat analisis baru atau pilih hasil analisis sebelumnya', 'error')
        return redirect(url_for('main.history'))

def history():
    session.pop('analysis_id', None)
    results = Analysis.query.filter_by(user_id=current_user.id).all()
    return render_template('main/history.html', results=results)

def delete_history(id):
    Analysis.query.filter_by(id=id).delete()
    db.session.commit()
    
    folder = os.path.join('app', 'static', 'img', 'wordclouds')
    models = ['svm', 'indobert']
    sentiments = ['positif','negatif']
    for model in models:
        for sentiment in sentiments:
            filename = folder+"\\"+sentiment+"\\"+model+"_"+id+".png"
            if os.path.exists(filename):
                os.remove(filename)
            
    return redirect(url_for('main.history'))

def get_result():
    session['analysis_id'] = request.form.get('id')
    return redirect(url_for('main.result'))

def get_data(mode):
    if mode == 'sentiment':
        _class = list(set(['positif', 'negatif', 'netral']) -
                        set([value[0] for value in db.session.query(Corpus.sentiment).filter(Corpus.sentiment_data!=True).distinct().all()]))
        text_1 = [value[0] for value in Corpus.query.filter(Corpus.sentiment_data!=True).with_entities(Corpus.prep_1).all()]
        text_2 = [value[0] for value in Corpus.query.filter(Corpus.sentiment_data!=True).with_entities(Corpus.prep_2).all()]
        label = [value[0] for value in Corpus.query.filter(Corpus.sentiment_data!=True).with_entities(Corpus.sentiment).all()]
    elif mode == 'intent':
        _class = list(set(['apresiasi', 'harapan', 'dukungan', 'sapaan', 'kritik', 'pertanyaan', 'keluhan', 'opini', 'saran', 'informasi', 'ujaran kebencian']) - 
                        set([value[0] for value in db.session.query(Corpus.intent).filter(Corpus.intent_data!=True).distinct().all()]))
        text_1 = [value[0] for value in Corpus.query.filter(Corpus.intent_data!=True).with_entities(Corpus.prep_1).all()]
        text_2 = [value[0] for value in Corpus.query.filter(Corpus.intent_data!=True).with_entities(Corpus.prep_2).all()]
        label = [value[0] for value in Corpus.query.filter(Corpus.intent_data!=True).with_entities(Corpus.intent).all()]

    return _class, text_1, text_2, label

def train_model():
    mode = request.form.get('mode')
    model = request.form.get('model')
    optimizer = request.form.get('optimizer')
    epochs = request.form.get('epochs')
    max_length = request.form.get('max_length')
    batch_size = request.form.get('batch_size')
    learning_rate = request.form.get('learning_rate')
    dropout_rate = request.form.get('dropout_rate')
    l2_reg = request.form.get('l2_reg')
    scheduler_warmup_steps = request.form.get('scheduler_warmup_steps')
    gradient_clip_value = request.form.get('gradient_clip_value')
    early_stopping_patience = request.form.get('early_stopping_patience')
    svm_kernel = request.form.get('svm_kernel')
    svm_c = request.form.get('svm_c')
    svm_gamma = request.form.get('svm_gamma')
    svm_class_weight = None if request.form.get('svm_class_weight') == 'None' else request.form.get('svm_class_weight')
    n_splits = request.form.get('n_splits')

    _class, text_1, text_2, label = get_data(mode)
    
    if _class:
        flash(f'Tambahkan {mode} {_class} untuk memulai pelatihan!', 'error')
        return redirect(url_for('words.corpus'))
    else:
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(text_1, label, test_size=0.2, random_state=42, stratify=label)
        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(text_2, label, test_size=0.2, random_state=42, stratify=label)

        indobert = IndoBERTFineTuner(mode, X_train_1, y_train_1, X_test_1, y_test_1, pretrained_model=model,
                    optimizer_name=optimizer, max_length=int(max_length), batch_size=int(batch_size),learning_rate=float(learning_rate),
                    epochs=int(epochs), dropout_rate=float(dropout_rate), l2_reg=float(l2_reg), early_stopping_patience=int(early_stopping_patience), 
                    scheduler_warmup_steps=int(scheduler_warmup_steps), gradient_clip_value=float(gradient_clip_value))
        svm = SVMModel(mode, X_train_2, X_test_2, y_train_2, y_test_2, n_splits=int(n_splits), svm_kernel=svm_kernel, 
                       svm_c=float(svm_c), svm_gamma=float(svm_gamma), svm_class_weight=svm_class_weight)
        try:            
            # indobert.training()
            svm.training()
            
            for data in X_train_1:
                corpus = Corpus.query.filter_by(prep_1=data).first()
                if mode == 'sentiment':
                    corpus.sentiment_data = True
                elif mode == 'intent':
                    corpus.intent_data = True
                db.session.commit()

            return redirect(url_for('main.train_result', mode=mode))            
        except Exception as e:
            flash(f'Gagal melatih model {e}', 'error')
            return redirect(url_for('words.corpus'))

def train_result(mode):
    try:
        title = f'hasil pelatihan model {mode}'
        indobert_path = 'img/charts/indobert/'+mode
        svm_path = 'img/charts/svm/'+mode
        cm1 = indobert_path+'/confusion_matrix.png'
        cm2 = svm_path+'/confusion_matrix.png'
        lc1 = indobert_path+'/learning_curve.png'
        lc2 = svm_path+'/learning_curve.png'
        report1 = joblib.load(os.path.join('app', 'ml', 'indobert', mode, 'classification_report.joblib'))
        report2 = joblib.load(os.path.join('app', 'ml', 'svm', mode, 'classification_report.joblib'))
        cr1 = pd.DataFrame(report1).transpose()    
        cr2 = pd.DataFrame(report2).transpose()
        
        return render_template("main/train_result.html", title=title, cm1=cm1, cm2=cm2, lc1=lc1, lc2=lc2, cr1=cr1, cr2=cr2)
    except FileNotFoundError:
        flash('Model belum dilatih!', 'error')
        return redirect(request.referrer)
    except Exception as e:
        flash('Ada masalah {e}!', 'error')
        return redirect(request.referrer)

from app.model import Informal
def tes():
    tokens = ['Aku', 'ingin', 'mandi']
    content = [Informal.query.filter_by(bentuk_informal=token).first().bentuk_formal if (informal:=Informal.query.filter_by(bentuk_informal=token).first()) else token for token in tokens]
    result = ' '.join(content)

    return render_template("tes.html", data=result)

