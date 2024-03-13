from app import db

class Youtube(db.Model):
    id = db.Column(db.String(100), primary_key=True)
    title = db.Column(db.String(100))
    publishedAt = db.Column(db.String(100))
    description = db.Column(db.Text)
    views = db.Column(db.Integer)
    likes = db.Column(db.Integer)
    favorites = db.Column(db.Integer)
    comments = db.Column(db.Integer)
    thumbnail = db.Column(db.String(255))

    def __repr__(self):
        return '{}'.format(self.title)  

    