import os
import random
from dotenv import load_dotenv

load_dotenv()

basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    HOST = str(os.environ.get('DB_HOST'))
    DATABASE = str(os.environ.get('DB_DATABASE'))
    USERNAME = str(os.environ.get('DB_USERNAME'))
    PASSWORD = str(os.environ.get('DB_PASSWORD'))
    
    JWT_SECRET_KEY = str(os.environ.get('JWT_SECRET'))
    SECRET_KEY = str(os.environ.get('SECRET_KEY'))
    SECURITY_PASSWORD_SALT = str(os.environ.get('SECURITY_PASSWORD_SALT'))
    
    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}/{DATABASE}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_RECORD_QUERIES = True
         
    UPLOAD_FOLDER = os.path.join('app', 'static', 'img')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  

    MAIL_SERVER = str(os.environ.get('SERVER_MAIL'))
    MAIL_PORT = str(os.environ.get('PORT_MAIL'))
    MAIL_USERNAME = str(os.environ.get('USERNAME_MAIL'))
    MAIL_PASSWORD = str(os.environ.get('PASSWORD_MAIL'))
    MAIL_DEFAULT_SENDER = str(os.environ.get('DEFAULT_SENDER_MAIL'))
    MAIL_USE_TLS = False
    MAIL_USE_SSL = True


    ADMIN_MAIL = str(os.environ.get('MAIL_ADMIN'))
    ADMIN_PASSWORD = str(os.environ.get('PASSWORD_ADMIN'))
    ADMIN_NAME = str(os.environ.get('NAME_ADMIN'))
    ADMIN_PHONE = str(os.environ.get('PHONE_ADMIN')) 

    RES_FOLDER = os.path.join('app', 'res')

    DEVELOPER_KEY1 = str(os.environ.get('DEV_KEY1'))
    DEVELOPER_KEY2 = str(os.environ.get('DEV_KEY2'))
    DEVELOPER_KEY3 = str(os.environ.get('DEV_KEY3'))

    DEVELOPER_KEYS = random.choice([DEVELOPER_KEY1, DEVELOPER_KEY2, DEVELOPER_KEY3])