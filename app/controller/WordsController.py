from app import app, db
from app.model import *
from app.helpers.commons import paginating, get_root_id
from app.helpers.classifying import textblob_classify, vader_classify, terjemahkan
from app.helpers.preprocessing import preprocessing_1, preprocessing_2

from flask import request, flash, redirect, url_for, render_template, session, make_response
from flask_login import current_user
# from flask_sqlalchemy_caching import FromCache
from sqlalchemy.exc import IntegrityError
import json
import os

#####################
##   AKAR KATA     ##
#####################
def root():
    title = 'kosakata dasar'
    result, pagination = paginating(Root, order_by=Root.akar_kata, search_within=Root.akar_kata)
    return render_template('kosakata/root.html', title=title, result=result, pagination=pagination)

def add_root():
    akar_kata = request.form.get('akar_kata').lower()
    root_word = request.form.get('root_word').lower() if request.form.get('root_word') != '' else terjemahkan(akar_kata)
    polaritas = textblob_classify(root_word)
    sumber = request.form.get('sumber')

    root = Root(
        akar_kata = akar_kata,
        root_word = root_word,        
        polaritas = polaritas,
        sumber = sumber,
        kontributor_id = current_user.id,
    )
    try:
        db.session.add(root)
        db.session.commit()
        flash("Berhasil menambah akar kata.", "success")
    except IntegrityError as e:
        db.session.rollback()
        error_info = e.orig.args
        if 'Duplicate entry' in str(error_info):
            flash('Duplikat entri: Akar kata yang sama sudah ada dalam database.', 'error')
        else:
            flash(f'Kesalahan IntegrityError: {error_info}', 'error')
    except Exception as e:
        print(e)
        flash("Gagal menambah akar kata.", "error")
    finally:
        return redirect(url_for('words.root'))    

def edit_root(id):
    try:
        root = Root.query.filter_by(id=id).first()
        root.akar_kata = request.form.get('akar_kata').lower()
        root.root_word = request.form.get('root_word').lower() if request.form.get('root_word') != '' else terjemahkan(root.akar_kata)
        root.polaritas = textblob_classify(root.root_word)
        root.sumber = request.form.get('sumber')
        root.editor_id = current_user.id
        db.session.commit()
        flash("Berhasil merubah akar kata.", "success")
    except Exception as e:
        print(e)
        flash("Gagal merubah akar kata.", "error")
    finally:
        return redirect(url_for('words.root'))
    
def delete_root(id):
    try:
        root = Root.query.filter_by(id=id).first()
        db.session.delete(root)
        db.session.commit()
        flash("Berhasil menghapus akar kata.", "success")
    except Exception as e:
        print(e)
        flash("Gagal menghapus akar kata.", "error")
    finally:
        return redirect(url_for('words.root'))


#####################
## KATA BERIMBUHAN ##
#####################
def compound():
    title = 'kosakata berimbuhan'
    roots_list = Root.query.order_by(Root.akar_kata).all()
    roots_list = list(map(lambda x: x.akar_kata, roots_list))

    result, pagination = paginating(Compound, order_by=Compound.root_id, search_within=Compound.kata_berimbuhan, search_within2=Root.akar_kata, join=Root)
    return render_template('kosakata/compound.html', title=title, result=result, pagination=pagination, data=json.dumps(roots_list))
    
def add_compound():
    kata_berimbuhan = request.form.get('kata_berimbuhan').lower()
    akar_kata = request.form.get('akar_kata').lower()
    sumber = request.form.get('sumber')
    root_word = terjemahkan(akar_kata)
    polaritas = textblob_classify(root_word)

    root = Root.query.filter_by(akar_kata=akar_kata).first()
    root_id = get_root_id(akar_kata, sumber)
    # if root:
    #     root_id = root.id
    # else:
    #     new_root = Root(
    #           akar_kata = akar_kata,
    #           sumber = sumber,
    #           root_word = root_word,
    #           polaritas = polaritas,
    #           kontributor_id = current_user.id,
    #       )
    #     db.session.add(new_root)
    #     db.session.commit()
    #     root_id = new_root.id    
    try:
        compound = Compound(
            kata_berimbuhan = kata_berimbuhan,
            root_id = root_id,
            sumber = sumber,
            kontributor_id = current_user.id,
        )
        
        db.session.add(compound)
        db.session.commit()
        flash("Berhasil menambah kata berimbuhan.", "success")
    except IntegrityError as e:
        db.session.rollback()
        error_info = e.orig.args
        if 'Duplicate entry' in str(error_info):
            flash('Duplikat entri: Kata berimbuhan yang sama sudah ada dalam database.', 'error')
        else:
            flash(f'Kesalahan IntegrityError: {error_info}', 'error')
    except Exception as e:
        print(e)
        flash("Gagal menambah kata berimbuhan.", "error")
    finally:
        return redirect(url_for('words.compound'))
    
def edit_compound(id):
    akar_kata = request.form.get('akar_kata').lower()
    sumber = request.form.get('sumber')

    root = Root.query.filter_by(akar_kata=akar_kata).first()
    if root:
        root_id = root.id
    else:
        try:
            new_root = Root(
                akar_kata = akar_kata,
                sumber = sumber,
                kontributor_id = current_user.id,
            )
            db.session.add(new_root)
            db.session.commit()
            root_id = new_root.id
        except Exception as e:
            print(e)
            flash("Gagal Menambah akar kata.", "error")
    try:
        compound = Compound.query.filter_by(id=id).first()
        compound.kata_berimbuhan = request.form.get('kata_berimbuhan').lower()
        compound.root_id = root_id
        compound.sumber = request.form.get('sumber')
        compound.editor_id = current_user.id
        db.session.commit()
        flash("Berhasil merubah kata berimbuhan.", "success")
    except Exception as e:
        print(e)
        flash("Gagal merubah kata berimbuhan.", "error")
    finally:
        return redirect(url_for('words.compound'))
    
def delete_compound(id):
    try:
        compound = Compound.query.filter_by(id=id).first()
        db.session.delete(compound)
        db.session.commit()
        flash("Berhasil menghapus kata berimbuhan.", "success")
    except Exception as e:
        print(e)
        flash("Gagal menghapus kata berimbuhan.", "error")
    finally:
        return redirect(url_for('words.compound'))
    

#####################
##  KATA INFORMAL  ##
#####################
def formal():
    title = 'kosakata informal'
    result, pagination = paginating(Informal, order_by=Informal.bentuk_informal, search_within=Informal.bentuk_informal, search_within2=Informal.bentuk_formal)
    return render_template('kosakata/formal.html', title=title, result=result, pagination=pagination)

def add_formal():
    word = Informal(
        bentuk_informal = request.form.get('bentuk_informal').lower(),
        bentuk_formal = request.form.get('bentuk_formal').lower(),
        sumber = request.form.get('sumber'),
        kontributor_id = current_user.id,
    )
    try:
        db.session.add(word)
        db.session.commit()
        flash("Berhasil menambah kata informal.", "success")
    except IntegrityError as e:
        db.session.rollback()
        error_info = e.orig.args
        if 'Duplicate entry' in str(error_info):
            flash('Duplikat entri: Kata informal yang sama sudah ada dalam database.', 'error')
        else:
            flash(f'Kesalahan IntegrityError: {error_info}', 'error')
    except Exception as e:
        print(e)
        flash("Gagal menambah kata informal.", "error")
    finally:
        return redirect(url_for('words.informal'))

def edit_formal(id):
    try:
        word = Informal.query.filter_by(id=id).first()
        word.bentuk_informal = request.form.get('bentuk_informal').lower()
        word.bentuk_formal = request.form.get('bentuk_formal').lower()
        word.sumber = request.form.get('sumber')
        word.editor_id = current_user.id
        db.session.commit()
        flash("Berhasil merubah kata informal.", "success")
    except Exception as e:
        print(e)
        flash("Gagal merubah kata informal.", "error")
    finally:
        return redirect(url_for('words.informal'))

def delete_formal(id):
    try:
        word = Informal.query.filter_by(id=id).first()
        db.session.delete(word)
        db.session.commit()
        flash("Berhasil menghapus kata informal.", "success")
    except Exception as e:
        print(e)
        flash("Gagal menghapus kata informal.", "error")
    finally:
        return redirect(url_for('words.informal'))
    

#####################
##   KATA NEGASI   ##
#####################
def negation():
    title = 'kosakata negasi'
    result, pagination = paginating(Negasi, order_by=Negasi.kata_negasi, search_within=Negasi.kata_negasi)
    return render_template('kosakata/negation.html', title=title, result=result, pagination=pagination)

def add_negation():
    word = Negasi(
        kata_negasi = request.form.get('kata_negasi').lower(),
        sumber = request.form.get('sumber'),
        kontributor_id = current_user.id,
    )
    try:
        db.session.add(word)
        db.session.commit()
        flash("Berhasil menambah kata negasi.", "success")
    except IntegrityError as e:
        db.session.rollback()
        error_info = e.orig.args
        if 'Duplicate entry' in str(error_info):
            flash('Duplikat entri: Kata negasi yang sama sudah ada dalam database.', 'error')
        else:
            flash(f'Kesalahan IntegrityError: {error_info}', 'error')
    except Exception as e:
        print(e)
        flash("Gagal menambah kata negasi.", "error")
    finally:
        return redirect(url_for('words.negation'))
    
def edit_negation(id):
    try:
        word = Negasi.query.filter_by(id=id).first()
        word.kata_negasi = request.form.get('kata_negasi').lower()
        word.sumber = request.form.get('sumber')
        word.editor_id = current_user.id
        db.session.commit()
        flash("Berhasil merubah kata negasi.", "success")
    except Exception as e:
        print(e)
        flash("Gagal merubah kata negasi.", "error")
    finally:
        return redirect(url_for('words.negation'))
    
def delete_negation(id):
    try:
        word = Negasi.query.filter_by(id=id).first()
        db.session.delete(word)
        db.session.commit()
        flash("Berhasil menghapus kata negasi.", "success")
    except Exception as e:
        print(e)
        flash("Gagal menghapus kata negasi.", "error")
    finally:
        return redirect(url_for('words.negation'))


#####################
##   STOPWORDS     ##
#####################    
def stopword():
    title = 'daftar stopword'
    result, pagination = paginating(Stopword, order_by=Stopword.stop_word, search_within=Stopword.stop_word)
    return render_template('kosakata/stopword.html', title=title, result=result, pagination=pagination)

def add_stopword():
    word = Stopword(
        stop_word = request.form.get('stop_word').lower(),
        sumber = request.form.get('sumber'),
        kontributor_id = current_user.id,
    )
    try:
        db.session.add(word)
        db.session.commit()
        flash("Berhasil menambah stopword.", "success")
    except IntegrityError as e:
        db.session.rollback()
        error_info = e.orig.args
        if 'Duplicate entry' in str(error_info):
            flash('Duplikat entri: Stopword yang sama sudah ada dalam database.', 'error')
        else:
            flash(f'Kesalahan IntegrityError: {error_info}', 'error')
    except Exception as e:
        print(e)
        flash("Gagal menambah stopword.", "error")
    finally:
        return redirect(url_for('words.stopword'))
    
def edit_stopword(id):
    try:
        word = Stopword.query.filter_by(id=id).first()
        word.stop_word = request.form.get('stop_word').lower()
        word.sumber = request.form.get('sumber')
        word.editor_id = current_user.id
        db.session.commit()
        flash("Berhasil merubah stopword.", "success")
    except Exception as e:
        print(e)
        flash("Gagal merubah stopword.", "error")
    finally:
        return redirect(url_for('words.stopword'))
    
def delete_stopword(id):
    try:
        word = Stopword.query.filter_by(id=id).first()
        db.session.delete(word)
        db.session.commit()
        flash("Berhasil menghapus stopword.", "success")
    except Exception as e:
        print(e)
        flash("Gagal menghapus stopword.", "error")
    finally:
        return redirect(url_for('words.stopword'))


#####################
##  KATA POSITIF   ##
#####################    
def positive():
    title = 'kosakata positif'
    result, pagination = paginating(Positive, order_by=Positive.kata_positif, search_within=Positive.kata_positif, search_within2=Positive.bobot)
    return render_template('kosakata/positive.html', title=title, result=result, pagination=pagination)

def add_positive():
    word = Positive(
        kata_positif = request.form.get('kata_positif').lower(),
        bobot = request.form.get('bobot'),
        sumber = request.form.get('sumber'),
        kontributor_id = current_user.id,
    )
    try:
        db.session.add(word)
        db.session.commit()
        flash("Berhasil menambah kata positif.", "success")
    except IntegrityError as e:
        db.session.rollback()
        error_info = e.orig.args
        if 'Duplicate entry' in str(error_info):
            flash('Duplikat entri: Kata positif yang sama sudah ada dalam database.', 'error')
        else:
            flash(f'Kesalahan IntegrityError: {error_info}', 'error')
    except Exception as e:
        print(e)
        flash("Gagal menambah kata positif.", "error")
    finally:
        return redirect(url_for('words.positive'))
    
def edit_positive(id):
    try:
        word = Positive.query.filter_by(id=id).first()
        word.kata_positif = request.form.get('kata_positif').lower()
        word.bobot = request.form.get('bobot')
        word.sumber = request.form.get('sumber')
        word.editor_id = current_user.id
        db.session.commit()
        flash("Berhasil merubah kata positif.", "success")
    except Exception as e:
        print(e)
        flash("Gagal merubah kata positif.", "error")
    finally:
        return redirect(url_for('words.positive'))
    
def delete_positive(id):
    try:
        word = Positive.query.filter_by(id=id).first()
        db.session.delete(word)
        db.session.commit()
        flash("Berhasil menghapus kata positif.", "success")
    except Exception as e:
        print(e)
        flash("Gagal menghapus kata positif.", "error")
    finally:
        return redirect(url_for('words.positive'))
    

#####################
##  KATA NEGATIF   ##
##################### 
def negative():
    title = 'kosakata negatif'
    result, pagination = paginating(Negative, order_by=Negative.kata_negatif, search_within=Negative.kata_negatif)
    return render_template('kosakata/negative.html', title=title, result=result, pagination=pagination)

def add_negative():
    word = Negative(
        kata_negatif = request.form.get('kata_negatif').lower(),
        bobot = request.form.get('bobot'),
        sumber = request.form.get('sumber'),
        kontributor_id = current_user.id,
    )
    try:
        db.session.add(word)
        db.session.commit()
        flash("Berhasil menambah kata negatif.", "success")
    except IntegrityError as e:
        db.session.rollback()
        error_info = e.orig.args
        if 'Duplicate entry' in str(error_info):
            flash('Duplikat entri: Kata Negatif yang sama sudah ada dalam database.', 'error')
        else:
            flash(f'Kesalahan IntegrityError: {error_info}', 'error')
    except Exception as e:
        print(e)
        flash("Gagal menambah kata negatif.", "error")
    finally:
        return redirect(url_for('words.negative'))
    
def edit_negative(id):
    try:
        word = Negative.query.filter_by(id=id).first()
        word.kata_negatif = request.form.get('kata_negatif').lower()
        word.bobot = request.form.get('bobot')
        word.sumber = request.form.get('sumber')
        word.editor_id = current_user.id
        db.session.commit()
        flash("Berhasil merubah kata negatif.", "success")
    except Exception as e:
        print(e)
        flash("Gagal merubah kata negatif.", "error")
    finally:
        return redirect(url_for('words.negative'))   

def delete_negative(id):
    try:
        word = Negative.query.filter_by(id=id).first()
        db.session.delete(word)
        db.session.commit()
        flash("Berhasil menghapus kata negatif.", "success")
    except Exception as e:
        print(e)
        flash("Gagal menghapus kata negatif.", "error")
    finally:
        return redirect(url_for('words.negative'))
    

#####################
##  CORPUS LATIH   ##
##################### 
def corpus():
    title = 'Korpus Latih'
    intent = list(set(['apresiasi', 'harapan', 'dukungan', 'sapaan', 'kritik', 'pertanyaan', 'keluhan', 'opini', 'saran', 'informasi', 'ujaran kebencian']) - 
                        set([value[0] for value in db.session.query(Corpus.intent).distinct().all()]))
    sentiment = list(set(['positif', 'negatif', 'netral']) -
                        set([value[0] for value in db.session.query(Corpus.sentiment).distinct().all()]))
    
    result, pagination = paginating(Corpus, order_by=Corpus.id, order='DESC', search_within=Corpus.text, filtering=Corpus.sentiment_data==False, filtering2=Corpus.intent_data==False)
    return render_template('kosakata/corpus.html', title=title, result=result, pagination=pagination, intent=intent, sentiment=sentiment)

def add_corpus():
    text = request.form.get('text').lower()
    prep_1 = preprocessing_1(text)
    prep_2 = preprocessing_2(prep_1)
        
    corpus = Corpus(
        text = text,
        prep_1 = prep_1,
        prep_2 = prep_2,
        sentiment = request.form.get('sentiment'),
        intent = request.form.get('intent'),
        kontributor_id = current_user.id,
    )
    try:
        db.session.add(corpus)
        db.session.commit()
        flash("Berhasil menambah data korpus.", "success")
    except IntegrityError as e:
        db.session.rollback()
        error_info = e.orig.args
        if 'Duplicate entry' in str(error_info):
            flash('Duplikat entri: Data korpus yang sama sudah ada dalam database.', 'error')
        else:
            flash(f'Kesalahan IntegrityError: {error_info}', 'error')
    except Exception as e:
        print(e)
        flash(f"Gagal menambah data: {e}.", "error")
    finally:
        return redirect(url_for('words.corpus'))

def edit_corpus(id):
    try:
        corpus = Corpus.query.filter_by(id=id).first()
        corpus.text = request.form.get('text').lower()
        corpus.prep_1 = request.form.get('prep_1').lower()
        corpus.prep_2 = preprocessing_2(corpus.prep_1)
        corpus.sentiment = request.form.get('sentiment')
        corpus.intent = request.form.get('intent')
        corpus.editor_id = current_user.id
        db.session.commit()
        flash("Berhasil merubah data korpus.", "success")
    except Exception as e:
        print(e)
        flash(f"Gagal merubah data korpus: {e}.", "error")
    finally:
        return redirect(url_for('words.corpus'))

def delete_corpus(id):
    try:
        corpus = Corpus.query.filter_by(id=id).first()
        db.session.delete(corpus)
        db.session.commit()
        flash("Berhasil menghapus data korpus.", "success")
    except Exception as e:
        print(e)
        flash(f"Gagal menghapus data korpus: {e}.", "error")
    finally:
        return redirect(url_for('words.corpus'))
    
