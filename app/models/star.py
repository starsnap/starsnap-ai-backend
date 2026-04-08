"""
Star 모델 - 스타 정보
"""
from db import db
from pgvector.sqlalchemy import Vector

class Star(db.Model):
    """
    스타 정보를 저장하는 모델
    
    Attributes:
        id (str): 고유한 스타 ID
        name (str): 스타 이름
        nickname (str): 스타 닉네임
        gender (str): 성별
        birthday (date): 생일
        explanation (str): 설명
        image_key (str): 이미지 키
        face_image_vector (Vector): 얼굴 임베딩 벡터
        star_group_id (str): 스타 그룹 ID
        created_at (datetime): 생성 시간
    """
    __tablename__ = 'star'
    
    id = db.Column(db.String(255), primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False)
    birthday = db.Column(db.Date)
    explanation = db.Column(db.String(500), nullable=False)
    face_image_vector = db.Column(Vector(512))
    gender = db.Column(db.String(255), nullable=False)
    image_key = db.Column(db.String(500), unique=True)
    name = db.Column(db.String(255), nullable=False)
    nickname = db.Column(db.String(255), nullable=False, unique=True)
    star_group_id = db.Column(db.String(255), db.ForeignKey('star_group.id'))

    def to_dict(self):
        """모델을 딕셔너리로 변환"""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'birthday': self.birthday.isoformat() if self.birthday else None,
            'explanation': self.explanation,
            'gender': self.gender,
            'image_key': self.image_key,
            'name': self.name,
            'nickname': self.nickname,
            'star_group_id': self.star_group_id
        }
