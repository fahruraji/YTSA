from app import db

class Processed(db.Model):
    id = db.Column(db.BigInteger, primary_key=True)
    comment_id = db.Column(db.BigInteger, db.ForeignKey('preprocessed.id', ondelete='CASCADE', onupdate='CASCADE'))
    comment = db.relationship('Preprocessed', backref='processed', uselist=False, lazy=True, foreign_keys=[comment_id])
    vectors = db.Column(db.Text)
    svm = db.Column(db.String(8))
    encodes = db.Column(db.Text)
    indobert = db.Column(db.String(8))
    intent = db.Column(db.String(20))
    feedback = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return '{}'.format(self.indobert)
    
    def to_print(self, type=None):
        if type == 'vector':
            return {
                'Komentar': 'comment',
                'Bobot TFIDF': 'vectors',
                'Encoding IndoBERT': 'encodes',
            }
        elif type == 'sentiment':
            return {
                'Komentar': 'comment',
                'Klasifikasi SVM': 'svm',
                'Klasifikasi IndoBERT': 'indobert',
                'Klasifikasi Intent': 'intent',
            }

    