from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_wtf import CSRFProtect
from flask_login import LoginManager
from flask_mail import Mail
import logging
import os
import datetime

logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.basicConfig(filename="flask.log", level=logging.WARNING)

db = SQLAlchemy()
migrate = Migrate()
csrf = CSRFProtect()
mail = Mail()

login_manager = LoginManager()
login_manager.session_protection = "strong"
login_manager.login_view = "auth.login"
login_manager.login_message = u"Silakan login untuk melanjutkan"
login_manager.login_message_category = "warning"

def clear_log(log_file):
   if os.path.exists(log_file):
      with open(log_file, 'w'):
         pass

def create_app():
   app = Flask(__name__)
   app.config.from_object(Config)

   clear_log("flask.log")

   db.init_app(app)
   migrate.init_app(app, db)
   csrf.init_app(app)
   mail.init_app(app)
   login_manager.init_app(app)

   return app

app = create_app()
app.app_context().push()