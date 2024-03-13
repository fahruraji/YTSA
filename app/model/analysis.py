from app import db
from datetime import datetime

class Analysis(db.Model):
    id = db.Column(db.BigInteger, primary_key=True)
    youtube_id = db.Column(db.String(100), db.ForeignKey('youtube.id'))
    youtube = db.relationship('Youtube', backref='analysis_youtube', uselist=False, lazy=True, foreign_keys=[youtube_id])
    user_id = db.Column(db.BigInteger, db.ForeignKey('user.id'))
    analyze_at = db.Column(db.DateTime, default=datetime.utcnow)
    comments = db.relationship('Comments', backref='analysis', cascade='all, delete')

    def __repr__(self):
        return '{}'.format(self.analyze_at)
    