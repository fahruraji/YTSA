from app import db

class Informal(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    bentuk_informal = db.Column(db.String(100), index=True, unique=True, nullable=False)
    bentuk_formal = db.Column(db.String(100), nullable=False)
    sumber = db.Column(db.String(200))
    kontributor_id = db.Column(db.BigInteger, db.ForeignKey('user.id'))
    kontributor = db.relationship('User', backref='informal_kontributor', uselist=False, lazy=True, foreign_keys=[kontributor_id])
    editor_id = db.Column(db.BigInteger, db.ForeignKey('user.id'))
    editor = db.relationship('User', backref='informal_editor', uselist=False, lazy=True, foreign_keys=[editor_id])

    def __repr__(self):
        return '{}'.format(self.bentuk_informal)
    
    def to_print(self, column1=None, column2=None):
        return {
            'Kata Informal': 'bentuk_informal',
            'Bentuk Formal': 'bentuk_formal',
            'Sumber': 'sumber',
            'Kontributor': 'kontributor',
            'Editor': 'editor',
        }
    