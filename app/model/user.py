from app import app, db
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin, db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    username = db.Column(db.String(100), index=True, unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    registered_on = db.Column(db.DateTime, default=datetime.utcnow)
    is_confirmed = db.Column(db.Boolean, default=False)
    confirmed_on = db.Column(db.DateTime, default=datetime.utcnow)
    nama = db.Column(db.String(100), nullable=False)
    jkel = db.Column(db.String(1), default='L')
    pekerjaan = db.Column(db.String(250))
    alamat = db.Column(db.Text)
    telepon = db.Column(db.String(20), nullable=False)
    foto = db.Column(db.String(255), default='profile-img/user.png')
    analysis = db.relationship('Analysis', backref='user', cascade='all, delete')
    
    def __repr__(self):
        return '{}'.format(self.nama)
    
    def set_password(self, password):
        self.password = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password, password)
    
    def create_admin(self):
        self.username = app.config['ADMIN_MAIL']
        self.set_password(app.config['ADMIN_PASSWORD'])
        self.nama = app.config['ADMIN_NAME']
        self.telepon = app.config['ADMIN_PHONE']
        self.is_admin = True
        self.is_confirmed = True
        self.confirmed_on = datetime.utcnow()
        self.foto = 'profile-img/admin.png'
        db.session.add(self)
        db.session.commit()