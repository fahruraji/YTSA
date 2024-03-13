from app import db

class Positive(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    kata_positif = db.Column(db.String(100), index=True, unique=True, nullable=False)
    bobot = db.Column(db.Integer)
    sumber = db.Column(db.String(200))
    kontributor_id = db.Column(db.BigInteger, db.ForeignKey('user.id'))
    kontributor = db.relationship('User', backref='positive_kontributor', uselist=False, lazy=True, foreign_keys=[kontributor_id])
    editor_id = db.Column(db.BigInteger, db.ForeignKey('user.id'))
    editor = db.relationship('User', backref='positive_editor', uselist=False, lazy=True, foreign_keys=[editor_id])

    def __repr__(self):
        return '{}'.format(self.kata_positif)
    
    def to_print(self, column1=None, column2=None):
        return {
            'Kata Positif': 'kata_positif',
            'Bobot': 'bobot',
            'Sumber': 'sumber',
            'Kontributor': 'kontributor',
            'Editor': 'editor',
        }
    