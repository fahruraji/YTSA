from app import db

class Preprocessed(db.Model):
    id = db.Column(db.BigInteger, primary_key=True)
    comment_id = db.Column(db.BigInteger, db.ForeignKey('comments.id', ondelete='CASCADE', onupdate='CASCADE'))
    comment = db.relationship('Comments', backref='preprocessed', uselist=False, lazy=True, foreign_keys=[comment_id])
    casefolded = db.Column(db.Text)
    tokenized = db.Column(db.Text)
    normalized = db.Column(db.Text)
    stemmed = db.Column(db.Text)
    filtered = db.Column(db.Text)

    def __repr__(self):
        return '{}'.format(self.stemmed)
    
    def to_print(self, column1=None, column2=None):
        return {
            column1.title(): column1,
            column2.title(): column2
        }
    