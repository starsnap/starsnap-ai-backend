"""
StarGroup 모델 - 스타 그룹 정보
"""
from db import db


class StarGroup(db.Model):
    """스타 그룹 최소 매핑 모델 (FK 해석용)."""
    __tablename__ = "star_group"

    id = db.Column(db.String(255), primary_key=True)

