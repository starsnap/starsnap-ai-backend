from __future__ import annotations

import os
import sys
import time
import types
import unittest
from unittest.mock import patch

import jwt
from flask import Blueprint, jsonify


_REQUIRED_ENV = {
    "DB_USER": "test",
    "DB_PASSWORD": "test",
    "DB_HOST": "localhost",
    "DB_PORT": "5432",
    "DB_NAME": "test",
    "DB_SCHEME": "postgresql",
    "DEBUG": "false",
    "ARCFACE_PROVIDERS": "CPUExecutionProvider",
    "ARCFACE_MODEL_NAME": "buffalo_l",
    "ARCFACE_DET_SIZE": "640",
    "MATCH_MIN_SIMILARITY": "0.45",
    "JWT_ACCESS_SECRET": "test-jwt-secret",
    "AI_INTERNAL_TOKEN": "test-internal-token",
    "PHOTO_API_URL": "https://example.invalid/api/file/photo",
    "PHOTO_API_TIMEOUT_SECONDS": "1",
    "ACCESS_LOG_ENABLED": "false",
}
for _key, _value in _REQUIRED_ENV.items():
    os.environ.setdefault(_key, _value)


from app import create_app  # noqa: E402
from app.utils.jwt_utils import _decode_access_token  # noqa: E402


class _SecurityTestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = "test-csrf-secret"
    WTF_CSRF_SECRET_KEY = "test-csrf-secret"
    WTF_CSRF_TIME_LIMIT = 3600
    WTF_CSRF_CHECK_DEFAULT = False
    SESSION_COOKIE_NAME = "starsnap-ai-csrf-test"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = False
    ACCESS_LOG_ENABLED = False


def _fake_route_modules():
    enroll_module = types.ModuleType("app.routes.enroll")
    enroll_bp = Blueprint("security_test_enroll", __name__, url_prefix="/api")

    @enroll_bp.post("/cookie-write")
    def cookie_write():
        return jsonify({"status": "ok"})

    enroll_module.embedding_service = object()
    enroll_module.enroll_bp = enroll_bp

    face_module = types.ModuleType("app.routes.face_analysis")
    face_bp = Blueprint("security_test_face", __name__, url_prefix="/api/internal/v1")

    @face_bp.post("/bearer-write")
    def bearer_write():
        return jsonify({"status": "ok"})

    face_module.face_analysis_bp = face_bp
    return enroll_module, face_module


class CsrfProtectionTest(unittest.TestCase):
    def setUp(self):
        enroll_module, face_module = _fake_route_modules()
        with patch.dict(
            sys.modules,
            {
                "app.routes.enroll": enroll_module,
                "app.routes.face_analysis": face_module,
            },
        ):
            self.app = create_app(_SecurityTestConfig)
        self.client = self.app.test_client()

    def test_cookie_authenticated_write_requires_csrf_token(self):
        self.client.set_cookie("access-token", "signed-upstream-token")
        missing = self.client.post("/api/cookie-write")
        self.assertEqual(missing.status_code, 400)

        token_response = self.client.get("/api/csrf-token")
        token = token_response.get_json()["csrfToken"]
        accepted = self.client.post(
            "/api/cookie-write",
            headers={"X-CSRFToken": token},
        )
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(token_response.headers["Cache-Control"], "no-store")

    def test_internal_bearer_blueprint_does_not_require_browser_csrf(self):
        response = self.client.post("/api/internal/v1/bearer-write")
        self.assertEqual(response.status_code, 200)


class JwtVerificationTest(unittest.TestCase):
    def test_wrong_signature_is_rejected_before_claims_are_used(self):
        token = jwt.encode(
            {"jti": "attacker", "authority": "ADMIN", "exp": int(time.time()) + 60},
            "wrong-secret",
            algorithm="HS256",
            headers={"JWT": "access"},
        )
        with self.assertRaises(jwt.InvalidSignatureError):
            _decode_access_token(token)

    def test_verified_token_with_wrong_type_is_rejected(self):
        token = jwt.encode(
            {"jti": "user-1", "authority": "ADMIN", "exp": int(time.time()) + 60},
            "test-jwt-secret",
            algorithm="HS256",
            headers={"JWT": "refresh"},
        )
        with self.assertRaisesRegex(ValueError, "Invalid token type"):
            _decode_access_token(token)


if __name__ == "__main__":
    unittest.main()
