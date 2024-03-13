from app import db

class Compound(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    kata_berimbuhan = db.Column(db.String(250), index=True, unique=True, nullable=False)
    root_id = db.Column(db.BigInteger, db.ForeignKey('root.id'))
    sumber = db.Column(db.String(200))
    kontributor_id = db.Column(db.BigInteger, db.ForeignKey('user.id'))
    kontributor = db.relationship('User', backref='compound_kontributor', uselist=False, lazy=True, foreign_keys=[kontributor_id])
    editor_id = db.Column(db.BigInteger, db.ForeignKey('user.id'))
    editor = db.relationship('User', backref='compound_editor', uselist=False, lazy=True, foreign_keys=[editor_id])

    def __repr__(self):
        return '{}'.format(self.kata_berimbuhan)
    
    def to_print(self, column1=None, column2=None):
        return {
            'Kata Berimbuhan': 'kata_berimbuhan',
            'Akar Kata': 'root',
            'Sumber': 'sumber',
            'Kontributor': 'kontributor',
            'Editor': 'editor',
        }
    