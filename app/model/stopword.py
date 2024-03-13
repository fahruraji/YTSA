from app import db

class Stopword(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    stop_word = db.Column(db.String(100), index=True, unique=True, nullable=False)
    sumber = db.Column(db.String(200))
    kontributor_id = db.Column(db.BigInteger, db.ForeignKey('user.id'))
    kontributor = db.relationship('User', backref='stopword_kontributor', uselist=False, lazy=True, foreign_keys=[kontributor_id])
    editor_id = db.Column(db.BigInteger, db.ForeignKey('user.id'))
    editor = db.relationship('User', backref='stopword_editor', uselist=False, lazy=True, foreign_keys=[editor_id])

    def __repr__(self):
        return '{}'.format(self.stop_word)
    
    def to_print(self, column1=None, column2=None):
        return {
            'Stopword': 'stop_word',
            'Sumber': 'sumber',
            'Kontributor': 'kontributor',
            'Editor': 'editor',
        }
    