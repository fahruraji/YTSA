from app import db

class Negasi(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    kata_negasi = db.Column(db.String(100), index=True, unique=True, nullable=False)
    sumber = db.Column(db.String(200))
    kontributor_id = db.Column(db.BigInteger, db.ForeignKey('user.id'))
    kontributor = db.relationship('User', backref='negasi_kontributor', uselist=False, lazy=True, foreign_keys=[kontributor_id])
    editor_id = db.Column(db.BigInteger, db.ForeignKey('user.id'))
    editor = db.relationship('User', backref='negasi_editor', uselist=False, lazy=True, foreign_keys=[editor_id])

    def __repr__(self):
        return '{}'.format(self.kata_negasi)
    
    def to_print(self, column1=None, column2=None):
        return {
            'Kata Negasi': 'kata_negasi',
            'Sumber': 'sumber',
            'Kontributor': 'kontributor',
            'Editor': 'editor',
        }
    