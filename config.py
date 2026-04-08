"""
설정 파일 - 환경 변수 로드
"""
import os
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER", "starsnap")
DB_PASSWORD = os.getenv("DB_PASSWORD", "wCapkqQPi8t3FZLGYgcGsdiQHJ11TqHom7g1IU6uSDZQ464OU6")
DB_HOST = os.getenv("DB_HOST", "starsnap-postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "starsnap")
DB_SCHEME = os.getenv("DB_SCHEME", "starsnap")

DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def _parse_providers(raw: str) -> list[str]:
    providers = [item.strip() for item in raw.split(",") if item.strip()]
    if not providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return providers

class Config:
    SQLALCHEMY_DATABASE_URI = DATABASE_URI
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DEBUG = True
    ARCFACE_PROVIDERS = _parse_providers(
        os.getenv("ARCFACE_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider")
    )
    ARCFACE_MODEL_NAME = os.getenv("ARCFACE_MODEL_NAME", "buffalo_l")
    ARCFACE_DET_SIZE = int(os.getenv("ARCFACE_DET_SIZE", "640"))
    # ArcFace buffalo_l 코사인 유사도 기준:
    # 같은 사람: 보통 0.30~0.60 (조명/각도/표정 차이에 따라)
    # 다른 사람: 보통 0.10~0.25
    # 권장 임계값: 0.35~0.45
    MATCH_MIN_SIMILARITY = float(os.getenv("MATCH_MIN_SIMILARITY", "0.4"))
