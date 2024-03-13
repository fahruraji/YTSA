from app import app, login_manager, db
from app.restriction import admin_required
from app.model import *
from app.helpers.commons import get_columns, get_root_id
from app.helpers.classifying import textblob_classify, terjemahkan
from app.helpers.preprocessing import preprocessing_1, preprocessing_2

from flask import request, Response, flash, redirect, session, url_for, render_template
from werkzeug.utils import secure_filename
from flask_login import login_required, current_user
from dask import dataframe as dd
import pandas as pd
import csv       
import os

from app.routes.auth import auth as auth_blueprint
app.register_blueprint(auth_blueprint)
from app.routes.main import main as main_blueprint
app.register_blueprint(main_blueprint)
from app.routes.user import user as user_blueprint
app.register_blueprint(user_blueprint)
from app.routes.words import words as words_blueprint
app.register_blueprint(words_blueprint)

@login_manager.user_loader
def loader_user(user_id):
    # return User.query.get(user_id)
    return db.session.get(User, user_id)

@app.errorhandler(403)
def forbidden(e):
  return render_template('error_page/forbidden.html'), 403

@app.errorhandler(404)
def not_found(e):
  return render_template('error_page/not_found.html'), 404

@app.errorhandler(500)
def server_error(e):
  return render_template('error_page/internal_error.html'), 500

@app.route('/<route>/print/<title>')
@app.route('/<route>/print/<title>/<type>')
@app.route('/<route>/print/<title>/<column1>/<column2>')
def print_table(route, title, type=None, column1=None, column2=None):
    model = eval(route.title())
    model = model()
    title = title.title()

    if 'analysis_id' in session:
        analysis_id = session['analysis_id']
    
    data = model.query
    if isinstance(model, Comments):
        data = data.filter(Comments.analysis_id==analysis_id).all()
    elif isinstance(model, Preprocessed):
        data = data.join(Comments).filter(Comments.analysis_id==analysis_id).all()
    elif isinstance(model, Processed):
        data = data.join(Preprocessed).join(Comments).filter(Comments.analysis_id==analysis_id).all()
    else:
        data = data.all()

    # to_skip = ['id','analysis_id']
    # columns = get_columns(model, to_skip)
    # columns = list(map(lambda x: x.replace('kontributor_id', 'kontributor'), columns))
    # columns = list(map(lambda x: x.replace('editor_id', 'editor'), columns))
    # headers = list(map(lambda x: x.replace('_', ' '), columns))
    
    if type == None:
      headers = list(model.to_print(column1, column2).keys())
      columns = list(model.to_print(column1, column2).values())
    else:
      headers = list(model.to_print(type).keys())
      columns = list(model.to_print(type).values())
 
    return render_template('print/table.html', title=title, headers=headers, data=data, columns=columns)

@app.route('/<route>/<blueprint>/import_csv', methods=['POST'])
@login_required
@admin_required
def import_csv(route, blueprint):
    if request.method == 'POST':
      name = request.form['name']
      file = request.files['file']
      file_extension = secure_filename(file.filename).split('.')[1]

      if name == '':
        file_name = secure_filename(file.filename)
      else:
        file_name = f"{name.lower()}.{file_extension}"

      file_path = os.path.join(app.config['RES_FOLDER'], file_name)

      if os.path.exists(file_path):
          flash('Ada file dengan nama sama. Silakan rename dulu!', 'error')
      elif file_extension != 'csv':
          flash('Gagal mengimpor data. File harus berekstensi .csv!', 'error')
      else:
          try:
            file.save(file_path)
            try:
              bulk_insert(eval(route.title()), file_path)
              flash('Berhasil mengimpor data.', 'success')
            except Exception as e:
              flash(f'Gagal mengimpor data! {e}', 'error')
          except Exception as e:
            flash(f'File gagal disimpan! {e}', 'error')

      return redirect(url_for(f'{blueprint}.{route}'))
    
def bulk_insert(destination, source):
    # df = dd.read_csv(source, delimiter=';')
    for df in pd.read_csv(source, delimiter=';', chunksize=10, encoding='utf-8'):        
      df = df.fillna("-")
      if 'kata_dasar' in df.columns:
        df['root_id'] = df['kata_dasar'].apply(get_root_id, sumber=df['sumber'][0])
        df['root_word'] = df['kata_dasar'].apply(terjemahkan)
        df['polaritas'] = df['root_word'].apply(textblob_classify)
      elif 'akar_kata' in df.columns:
        df['root_word'] = df['akar_kata'].apply(terjemahkan)
        df['polaritas'] = df['root_word'].apply(textblob_classify)
      elif 'text' in df.columns:
        df['prep_1'] = df['text'].apply(preprocessing_1)
        df['prep_2'] = df['prep_1'].apply(preprocessing_2)
        df['sentiment_data'] = 0
        df['intent_data'] = 0
      to_skip = ['id', 'kontributor_id', 'editor_id', 'trained']
      cols = get_columns(destination, to_skip)
      data = []
      for index, row in df.iterrows():
          object = { col : row[col] for col in cols }
          object['kontributor_id'] = current_user.id
          
          data.append(object)

          db.session.execute(destination.__table__
              .insert()
              .prefix_with('IGNORE')
              .values(data))
          db.session.commit()


# def get_root_id(akar_kata, sumber):
#     try:
#         root = Root.query.filter_by(akar_kata=akar_kata).first()
#         root_id = root.id
#     except:
#         root_word = terjemahkan(akar_kata)
#         polaritas = textblob_classify(root_word)
#         new_root = Root(
#               akar_kata = akar_kata,
#               sumber = sumber,
#               root_word = root_word,
#               polaritas = polaritas,
#               kontributor_id = current_user.id,
#           )
#         db.session.add(new_root)
#         db.session.commit()
#         root_id = new_root.id

#     return root_id
    
@app.route('/<route>/template')
@login_required
@admin_required
def template(route):
    model = eval(route.title())
    to_skip = ['id', 'root_word', 'polaritas', 'kontributor_id', 'editor_id', 'sentiment_data', 'intent_data', 'prep_1', 'prep_2']
    cols = get_columns(model, to_skip)
    cols = list(map(lambda x: x.replace('root_id', 'kata_dasar'), cols))
    csv = ';'.join([str(elem) for elem in cols])
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                "attachment; filename="+route+".csv"})