"""
설정 파일 - dotenv + services.yaml + 환경변수 우선순위 로드
"""
from __future__ import annotations

import os
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


# DB 관련 필수 설정
DB_USER = _require_env("DB_USER")
DB_PASSWORD = _require_env("DB_PASSWORD")
DB_HOST = _require_env("DB_HOST")
DB_PORT = _require_env("DB_PORT")
DB_NAME = _require_env("DB_NAME")
DB_SCHEME = _require_env("DB_SCHEME")

DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


class Config:
    SQLALCHEMY_DATABASE_URI = DATABASE_URI
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DEBUG = _require_bool_env("DEBUG")
    ARCFACE_PROVIDERS = _parse_providers(_require_env("ARCFACE_PROVIDERS"))
    ARCFACE_MODEL_NAME = _require_env("ARCFACE_MODEL_NAME")
    ARCFACE_DET_SIZE = _require_int_env("ARCFACE_DET_SIZE")
    # 얼굴 검출용 입력 이미지 최대 긴 변 길이 (큰 이미지는 자동 축소)
    ARCFACE_MAX_IMAGE_DIM = int(os.getenv("ARCFACE_MAX_IMAGE_DIM", "1280"))
    # ArcFace 기준 설명
    MATCH_MIN_SIMILARITY = _require_float_env("MATCH_MIN_SIMILARITY")

    # JWT
    JWT_ACCESS_SECRET = _require_env("JWT_ACCESS_SECRET")

    # PHOTO API: env 우선, 없으면 services.yaml에 정의된 값을 사용
    PHOTO_API_URL = _require_env("PHOTO_API_URL")
    PHOTO_API_TIMEOUT_SECONDS = _require_float_env("PHOTO_API_TIMEOUT_SECONDS")

    # Access Log Forwarder
    ACCESS_LOG_ENABLED: bool = os.getenv("ACCESS_LOG_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
    ACCESS_LOG_URL: str = os.getenv("ACCESS_LOG_URL", "http://host.docker.internal:7070/api/server-logs")
    ACCESS_LOG_SERVICE_NAME: str = os.getenv("ACCESS_LOG_SERVICE_NAME", "starsnap-ai-backend")
