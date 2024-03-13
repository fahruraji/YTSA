from app import db

class Root(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    akar_kata = db.Column(db.String(150), index=True, unique=True, nullable=False)
    root_word = db.Column(db.String(150))
    polaritas = db.Column(db.String(8))
    sumber = db.Column(db.String(200))
    kontributor_id = db.Column(db.BigInteger, db.ForeignKey('user.id'))
    kontributor = db.relationship('User', foreign_keys=[kontributor_id], backref='root_kontributor')
    editor_id = db.Column(db.BigInteger, db.ForeignKey('user.id'))
    editor = db.relationship('User', foreign_keys=[editor_id], backref='root_editor')

    compound = db.relationship('Compound', backref='root', cascade='all, delete')
    def __repr__(self):
        return '{}'.format(self.akar_kata)
    
    def to_print(self, column1=None, column2=None):
        return {
            'Kata Dasar': 'akar_kata',
            'In English': 'root_word',
            'Polaritas': 'polaritas',
            'Sumber': 'sumber'
        }
    