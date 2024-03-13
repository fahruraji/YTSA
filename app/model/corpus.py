from app import db

class Corpus(db.Model):
    id = db.Column(db.BigInteger, primary_key=True)
    text = db.Column(db.Text, index=True, unique=True, nullable=False)
    prep_1 = db.Column(db.Text)
    prep_2 = db.Column(db.Text)
    sentiment = db.Column(db.String(8))
    intent = db.Column(db.String(20))
    sentiment_data = db.Column(db.Boolean, default=False)
    intent_data = db.Column(db.Boolean, default=False)
    kontributor_id = db.Column(db.BigInteger, db.ForeignKey('user.id'))
    kontributor = db.relationship('User', backref='feedback_kontributor', uselist=False, lazy=True, foreign_keys=[kontributor_id])
    editor_id = db.Column(db.BigInteger, db.ForeignKey('user.id'))
    editor = db.relationship('User', backref='feedback_editor', uselist=False, lazy=True, foreign_keys=[editor_id])

    def __repr__(self):
        return '{}'.format(self.id)

    