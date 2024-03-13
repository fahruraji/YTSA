from app import app, mail, db
from flask import request
from flask_paginate import Pagination, get_page_args, get_page_parameter
from app.helpers.classifying import terjemahkan, textblob_classify
import itsdangerous
from itsdangerous import URLSafeTimedSerializer
from flask_login import current_user
from flask_mail import Message
from sqlalchemy import or_, and_
from app.model import Root

def paginating(source, order_by=None, order=None, search_within=None, search_within2=None, join=None, filtering=None, filtering2=None, rows_per_page=10):
    page = request.args.get(get_page_parameter(), type=int, default=1)
    search = request.args.get('search')

    result = source.query

    if not order_by is None:        
        if not order is None:
            if order == 'DESC':
                result = result.order_by(order_by.desc())
        else:
            result = result.order_by(order_by)


    if not filtering is None and not filtering2 is None:
        result = result.filter(or_(filtering, filtering2))
    elif not filtering is None:
        result = result.filter(filtering)    

    if search:
        if search_within2:
            if join:
                result = result.join(join).filter(or_(search_within.ilike(f'%{search}%'), search_within2.ilike(f'%{search}%')))
            else:
                result = result.filter(or_(search_within.ilike(f'%{search}%'), search_within2.ilike(f'%{search}%')))
        else:
            result = result.filter(search_within.like(f'%{search}%'))

    result = result.paginate(page=page, per_page=rows_per_page)
        
    pagination = Pagination(page=page, per_page=rows_per_page, total=result.total, css_framework='bootstrap4')
    return result, pagination

def generate_confirmation_token(email):
    serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    return serializer.dumps(email, salt=app.config['SECURITY_PASSWORD_SALT'])


from itsdangerous import URLSafeTimedSerializer

def confirm_token(token, expiration=3600):
    serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    try:
        email = serializer.loads(
            token,
            salt=app.config['SECURITY_PASSWORD_SALT'],
            max_age=expiration
        )
        return email, False
    except itsdangerous.exc.SignatureExpired:
        email = serializer.loads(
            token,
            salt=app.config['SECURITY_PASSWORD_SALT']
        )
        new_token = generate_confirmation_token(email)
        return email, new_token
    except Exception as e:
        print(e)
        return False

    


def send_mail(to, subject, template):
    msg = Message(
        subject,
        recipients=[to],
        html=template,
        sender=app.config['MAIL_DEFAULT_SENDER']
    )
    mail.send(msg)


def get_columns(model, to_skip=None):
    if to_skip is None:
        return [column.key for column in model.__mapper__.columns]
    else:
        return [column.key for column in model.__mapper__.columns if column.key not in to_skip]
    

def get_root_id(akar_kata, sumber):
    try:
        root = Root.query.filter_by(akar_kata=akar_kata).first()
        root_id = root.id
    except:
        root_word = terjemahkan(akar_kata)
        polaritas = textblob_classify(root_word)
        new_root = Root(
              akar_kata = akar_kata,
              sumber = sumber,
              root_word = root_word,
              polaritas = polaritas,
              kontributor_id = current_user.id,
          )
        db.session.add(new_root)
        db.session.commit()
        root_id = new_root.id

    return root_id