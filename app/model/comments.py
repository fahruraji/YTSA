from app import db

class Comments(db.Model):
    id = db.Column(db.BigInteger, primary_key=True)
    analysis_id = db.Column(db.BigInteger, db.ForeignKey('analysis.id', ondelete='CASCADE', onupdate='CASCADE'))
    title = db.Column(db.String(100))
    name = db.Column(db.String(100), nullable=False)
    comment = db.Column(db.Text, nullable=False)
    published_at = db.Column(db.DateTime, nullable=False)
    likes = db.Column(db.Integer)
    replies = db.Column(db.Integer)

    def __repr__(self):
        return '{}'.format(self.comment)
    
    def to_print(self, column1=None, column2=None):
        return {
            'Nama Pengguna': 'name',
            'Isi Komentar': 'comment',
            'Diposting': 'published_at',
            'Menyukai': 'likes',
            'Membalas': 'replies'
        }
    