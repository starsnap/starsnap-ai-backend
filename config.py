"""
설정 파일 - dotenv + services.yaml + 환경변수 우선순위 로드
"""
from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml

load_dotenv()


def _load_services_file(path: str | None = None) -> dict[str, Any]:
    services_path = path or os.getenv("SERVICES_FILE", "services.yaml")
    p = Path(services_path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


def _parse_providers(raw: str) -> list[str]:
    providers = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return providers


# services.yaml -> env 매핑 (env가 이미 있으면 덮어쓰지 않음)
_SERVICES = _load_services_file()
_SERVICE_KEY_MAP = {
    "photo_api": "PHOTO_API_URL",
    "photo_api_timeout": "PHOTO_API_TIMEOUT_SECONDS",
}
for svc_key, env_key in _SERVICE_KEY_MAP.items():
    if env_key not in os.environ and svc_key in _SERVICES:
        os.environ[env_key] = str(_SERVICES[svc_key])


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def _require_env_or_file(value_key: str, file_key: str) -> str:
    direct_value = os.getenv(value_key)
    if direct_value is not None and direct_value.strip():
        return direct_value

    configured_path = os.getenv(file_key)
    if configured_path is None or not configured_path.strip():
        raise RuntimeError(
            f"Missing required secret: set {value_key} or {file_key}"
        )

    try:
        file_value = Path(configured_path.strip()).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as e:
        raise RuntimeError(
            f"Could not read secret file configured by {file_key}"
        ) from e
    if not file_value:
        raise RuntimeError(f"Secret file configured by {file_key} must not be empty")
    return file_value


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


def _require_similarity_env(key: str) -> float:
    value = _require_float_env(key)
    if not math.isfinite(value) or value < -1.0 or value > 1.0:
        raise RuntimeError(
            f"Environment variable {key} must be a finite number between -1 and 1, got: {value}"
        )
    return value


def _require_bool_env(key: str) -> bool:
    raw = _require_env(key).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Environment variable {key} must be bool, got: {raw}")


def _bool_env_with_default(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Environment variable {key} must be bool, got: {raw}")


def _int_env_with_default(key: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(key, str(default))
    try:
        value = int(raw)
    except ValueError as e:
        raise RuntimeError(f"Environment variable {key} must be int, got: {raw}") from e
    if value < minimum:
        raise RuntimeError(f"Environment variable {key} must be >= {minimum}, got: {raw}")
    return value


# Stateless 얼굴 분석 모드에서는 운영 DB를 초기화하거나 연결하지 않는다.
_DATABASE_ENABLED = _bool_env_with_default("AI_DATABASE_ENABLED", True)
if _DATABASE_ENABLED:
    DB_USER = _require_env("DB_USER")
    DB_PASSWORD = _require_env("DB_PASSWORD")
    DB_HOST = _require_env("DB_HOST")
    DB_PORT = _require_env("DB_PORT")
    DB_NAME = _require_env("DB_NAME")
    DB_SCHEME = _require_env("DB_SCHEME")
    DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    # Flask-SQLAlchemy는 비활성 상태에서 초기화되지 않는다. 실수로 다시
    # 초기화 코드가 추가되더라도 임시 DB로 조용히 우회하지 않도록 fail-closed 한다.
    DATABASE_URI = None


class Config:
    AI_DATABASE_ENABLED = _DATABASE_ENABLED
    SQLALCHEMY_DATABASE_URI = DATABASE_URI
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DEBUG = _require_bool_env("DEBUG")
    ARCFACE_PROVIDERS = _parse_providers(_require_env("ARCFACE_PROVIDERS"))
    ARCFACE_MODEL_NAME = _require_env("ARCFACE_MODEL_NAME")
    ARCFACE_DET_SIZE = _require_int_env("ARCFACE_DET_SIZE")
    # 얼굴 검출용 입력 이미지 최대 긴 변 길이 (큰 이미지는 자동 축소)
    ARCFACE_MAX_IMAGE_DIM = int(os.getenv("ARCFACE_MAX_IMAGE_DIM", "1280"))
    # ArcFace 기준 설명
    MATCH_MIN_SIMILARITY = _require_similarity_env("MATCH_MIN_SIMILARITY")

    # JWT
    JWT_ACCESS_SECRET = _require_env("JWT_ACCESS_SECRET")

    # 브라우저 쿠키 인증 API의 CSRF 보호. 별도 키가 없으면 기존 JWT secret을
    # 사용해 배포 호환성을 유지하되, 운영에서는 독립 키 사용을 권장한다.
    CSRF_SECRET_KEY = (
        os.getenv("CSRF_SECRET_KEY", JWT_ACCESS_SECRET).strip()
        or JWT_ACCESS_SECRET
    )
    SECRET_KEY = CSRF_SECRET_KEY
    WTF_CSRF_SECRET_KEY = CSRF_SECRET_KEY
    WTF_CSRF_TIME_LIMIT = 3600
    WTF_CSRF_CHECK_DEFAULT = False
    SESSION_COOKIE_NAME = "starsnap-ai-csrf"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = _bool_env_with_default(
        "SESSION_COOKIE_SECURE",
        not DEBUG,
    )

    # 메인 백엔드 -> AI 백엔드 전용 내부 API 인증/제한
    AI_INTERNAL_TOKEN = _require_env_or_file(
        "AI_INTERNAL_TOKEN",
        "AI_INTERNAL_TOKEN_FILE",
    )
    AI_FACE_ANALYSIS_MAX_IMAGE_BYTES = _int_env_with_default(
        "AI_FACE_ANALYSIS_MAX_IMAGE_BYTES",
        15 * 1024 * 1024,
    )
    AI_FACE_ANALYSIS_MAX_PIXELS = _int_env_with_default(
        "AI_FACE_ANALYSIS_MAX_PIXELS",
        60_000_000,
    )
    AI_FACE_ANALYSIS_MAX_FACES = _int_env_with_default(
        "AI_FACE_ANALYSIS_MAX_FACES",
        10,
    )
    AI_FACE_ANALYSIS_MATCH_STARS = _bool_env_with_default(
        "AI_FACE_ANALYSIS_MATCH_STARS",
        True,
    )
    AI_FACE_MODEL_VERSION = (
        os.getenv("AI_FACE_MODEL_VERSION", "insightface-0.7.3").strip()
        or "insightface-0.7.3"
    )

    # PHOTO API: env 우선, 없으면 services.yaml에 정의된 값을 사용
    PHOTO_API_URL = _require_env("PHOTO_API_URL")
    PHOTO_API_TIMEOUT_SECONDS = _require_float_env("PHOTO_API_TIMEOUT_SECONDS")

    # Access Log Forwarder
    ACCESS_LOG_ENABLED: bool = os.getenv("ACCESS_LOG_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
    ACCESS_LOG_URL: str = os.getenv("ACCESS_LOG_URL", "http://host.docker.internal:8081/api/server-logs")
    ACCESS_LOG_SERVICE_NAME: str = os.getenv("ACCESS_LOG_SERVICE_NAME", "starsnap-ai-backend")
    ACCESS_LOG_SECRET: str = os.getenv("ACCESS_LOG_SECRET", "").strip()
