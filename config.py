"""
설정 파일 - 환경 변수 로드
"""
import os
from dotenv import load_dotenv

load_dotenv()

def _require_env(key: str) -> str:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def _require_int_env(key: str) -> int:
    raw = _require_env(key)
    try:
        return int(raw)
    except ValueError as e:
        raise RuntimeError(f"Environment variable {key} must be int, got: {raw}") from e


def _require_float_env(key: str) -> float:
    raw = _require_env(key)
    try:
        return float(raw)
    except ValueError as e:
        raise RuntimeError(f"Environment variable {key} must be float, got: {raw}") from e


def _require_bool_env(key: str) -> bool:
    raw = _require_env(key).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Environment variable {key} must be bool, got: {raw}")


DB_USER = _require_env("DB_USER")
DB_PASSWORD = _require_env("DB_PASSWORD")
DB_HOST = _require_env("DB_HOST")
DB_PORT = _require_env("DB_PORT")
DB_NAME = _require_env("DB_NAME")
DB_SCHEME = _require_env("DB_SCHEME")

DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def _parse_providers(raw: str) -> list[str]:
    providers = [item.strip() for item in raw.split(",") if item.strip()]
    if not providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return providers

class Config:
    SQLALCHEMY_DATABASE_URI = DATABASE_URI
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DEBUG = _require_bool_env("DEBUG")
    ARCFACE_PROVIDERS = _parse_providers(
        _require_env("ARCFACE_PROVIDERS")
    )
    ARCFACE_MODEL_NAME = _require_env("ARCFACE_MODEL_NAME")
    ARCFACE_DET_SIZE = _require_int_env("ARCFACE_DET_SIZE")
    # ArcFace buffalo_l 코사인 유사도 기준:
    # 같은 사람: 보통 0.30~0.60 (조명/각도/표정 차이에 따라)
    # 다른 사람: 보통 0.10~0.25
    # 권장 임계값: 0.35~0.45
    MATCH_MIN_SIMILARITY = _require_float_env("MATCH_MIN_SIMILARITY")

    # JWT 설정 (starsnap-backend와 동일한 시크릿 키)
    JWT_ACCESS_SECRET = _require_env("JWT_ACCESS_SECRET")
