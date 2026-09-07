from __future__ import annotations

import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from flask import Blueprint, Flask


_REQUIRED_ENV = {
    "DB_USER": "test",
    "DB_PASSWORD": "test",
    "DB_HOST": "127.0.0.1",
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
    "PHOTO_API_URL": "http://127.0.0.1:8080/api/file/photo",
    "PHOTO_API_TIMEOUT_SECONDS": "1",
}
for _key, _value in _REQUIRED_ENV.items():
    os.environ.setdefault(_key, _value)

from app import create_app  # noqa: E402
from app.routes.face_analysis import face_analysis_bp  # noqa: E402
from config import _bool_env_with_default, _require_env_or_file  # noqa: E402


def _embedding(index: int) -> np.ndarray:
    value = np.zeros(512, dtype=np.float32)
    value[index] = 1.0
    return value


class _FakeEmbeddingService:
    active_providers = ["CPUExecutionProvider"]

    def __init__(self, analysis):
        self.analysis = analysis
        self.extract_calls = 0
        self.index_calls = 0
        self.match_calls = 0

    def extract_face_embeddings(self, _content, max_faces):
        self.extract_calls += 1
        self.requested_max_faces = max_faces
        return self.analysis

    def load_star_embedding_index(self):
        self.index_calls += 1
        return {"star_ids": ["star-1"], "embeddings": np.vstack([_embedding(0)])}

    def find_best_star_matches(self, embeddings, star_index, min_similarity):
        self.match_calls += 1
        self.match_args = (embeddings, star_index, min_similarity)
        return [{"starId": "star-1", "similarity": 0.91}] + [None] * (len(embeddings) - 1)


def _analysis(faces):
    return {
        "width": 1920,
        "height": 1080,
        "detected_face_count": len(faces),
        "processed_face_count": len(faces),
        "truncated": False,
        "faces": faces,
    }


def _make_app(service, max_bytes=1024, match_stars=True):
    app = Flask(__name__)
    app.config.update(
        TESTING=True,
        AI_INTERNAL_TOKEN="internal-secret",
        AI_FACE_ANALYSIS_MAX_IMAGE_BYTES=max_bytes,
        AI_FACE_ANALYSIS_MAX_FACES=10,
        AI_FACE_ANALYSIS_MATCH_STARS=match_stars,
        AI_FACE_MODEL_VERSION="insightface-0.7.3",
        ARCFACE_MODEL_NAME="buffalo_l",
        MATCH_MIN_SIMILARITY=0.45,
    )
    app.extensions["embedding_service"] = service
    app.register_blueprint(face_analysis_bp)
    return app


class FaceAnalysisRouteTest(unittest.TestCase):
    def test_internal_token_direct_env_takes_precedence_over_file(self):
        with patch.dict(
            os.environ,
            {
                "AI_INTERNAL_TOKEN_TEST": "direct-secret",
                "AI_INTERNAL_TOKEN_FILE_TEST": "missing-secret-file",
            },
            clear=False,
        ):
            self.assertEqual(
                "direct-secret",
                _require_env_or_file(
                    "AI_INTERNAL_TOKEN_TEST",
                    "AI_INTERNAL_TOKEN_FILE_TEST",
                ),
            )

    def test_internal_token_falls_back_to_trimmed_file_value(self):
        with tempfile.TemporaryDirectory() as directory:
            secret_path = Path(directory) / "internal-token"
            secret_path.write_text("\n  file-secret  \r\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "AI_INTERNAL_TOKEN_TEST": "   ",
                    "AI_INTERNAL_TOKEN_FILE_TEST": str(secret_path),
                },
                clear=False,
            ):
                self.assertEqual(
                    "file-secret",
                    _require_env_or_file(
                        "AI_INTERNAL_TOKEN_TEST",
                        "AI_INTERNAL_TOKEN_FILE_TEST",
                    ),
                )

    def test_internal_token_fails_fast_when_sources_are_missing_or_empty(self):
        value_key = "AI_INTERNAL_TOKEN_MISSING_TEST"
        file_key = "AI_INTERNAL_TOKEN_FILE_MISSING_TEST"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(value_key, None)
            os.environ.pop(file_key, None)
            with self.assertRaisesRegex(RuntimeError, "Missing required secret"):
                _require_env_or_file(value_key, file_key)

        with tempfile.TemporaryDirectory() as directory:
            secret_path = Path(directory) / "empty-internal-token"
            secret_path.write_text(" \r\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {value_key: "", file_key: str(secret_path)},
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "must not be empty"):
                    _require_env_or_file(value_key, file_key)

    def test_match_stars_config_defaults_true_and_parses_false(self):
        key = "AI_FACE_ANALYSIS_MATCH_STARS_TEST"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(key, None)
            self.assertTrue(_bool_env_with_default(key, True))
            os.environ[key] = "false"
            self.assertFalse(_bool_env_with_default(key, True))

    def test_requires_internal_bearer_token_with_json_error(self):
        service = _FakeEmbeddingService(_analysis([]))
        client = _make_app(service).test_client()

        response = client.post("/api/internal/v1/face-analysis")

        self.assertEqual(401, response.status_code)
        self.assertEqual("unauthorized", response.get_json()["error"]["code"])
        self.assertEqual(response.get_json()["requestId"], response.headers["X-Request-Id"])
        self.assertEqual(0, service.extract_calls)

    def test_non_ascii_bearer_token_is_rejected_as_json_not_server_error(self):
        service = _FakeEmbeddingService(_analysis([]))
        client = _make_app(service).test_client()

        response = client.post(
            "/api/internal/v1/face-analysis",
            headers={"Authorization": "Bearer 잘못된토큰"},
        )

        self.assertEqual(401, response.status_code)
        self.assertEqual("unauthorized", response.get_json()["error"]["code"])

    def test_returns_multi_face_contract_and_reuses_one_star_index(self):
        faces = [
            {"face_index": 0, "bbox": [10, 20, 30, 40], "confidence": 0.99, "embedding": _embedding(0)},
            {"face_index": 1, "bbox": [50, 60, 70, 80], "confidence": 0.88, "embedding": _embedding(1)},
        ]
        service = _FakeEmbeddingService(_analysis(faces))
        client = _make_app(service).test_client()

        response = client.post(
            "/api/internal/v1/face-analysis",
            data={"file": (io.BytesIO(b"image"), "snap.jpg")},
            headers={
                "Authorization": "Bearer internal-secret",
                "X-Request-Id": "snap-request-123",
            },
            content_type="multipart/form-data",
        )

        body = response.get_json()
        self.assertEqual(200, response.status_code)
        self.assertEqual("1", body["schemaVersion"])
        self.assertEqual("snap-request-123", body["requestId"])
        self.assertEqual("snap-request-123", response.headers["X-Request-Id"])
        self.assertEqual({"width": 1920, "height": 1080}, body["image"])
        self.assertEqual(2, body["detectedFaceCount"])
        self.assertEqual(2, body["processedFaceCount"])
        self.assertEqual("star-1", body["faces"][0]["bestMatch"]["starId"])
        self.assertIsNone(body["faces"][1]["bestMatch"])
        self.assertEqual(512, len(body["faces"][0]["embedding"]))
        self.assertEqual(1, service.index_calls)
        self.assertEqual(1, service.match_calls)

    def test_stateless_mode_returns_embeddings_without_querying_star_index(self):
        faces = [
            {"face_index": 0, "bbox": [10, 20, 30, 40], "confidence": 0.99, "embedding": _embedding(0)},
            {"face_index": 1, "bbox": [50, 60, 70, 80], "confidence": 0.88, "embedding": _embedding(1)},
        ]
        service = _FakeEmbeddingService(_analysis(faces))
        client = _make_app(service, match_stars=False).test_client()

        response = client.post(
            "/api/internal/v1/face-analysis",
            data={"file": (io.BytesIO(b"image"), "snap.jpg")},
            headers={"Authorization": "Bearer internal-secret"},
            content_type="multipart/form-data",
        )

        body = response.get_json()
        self.assertEqual(200, response.status_code)
        self.assertEqual(2, body["processedFaceCount"])
        self.assertEqual(512, len(body["faces"][0]["embedding"]))
        self.assertTrue(all(face["bestMatch"] is None for face in body["faces"]))
        self.assertEqual(0, service.index_calls)
        self.assertEqual(0, service.match_calls)

    def test_no_face_is_success_and_does_not_query_stars(self):
        service = _FakeEmbeddingService(_analysis([]))
        client = _make_app(service).test_client()
        response = client.post(
            "/api/internal/v1/face-analysis",
            data={"file": (io.BytesIO(b"image"), "snap.jpg")},
            headers={"Authorization": "Bearer internal-secret"},
            content_type="multipart/form-data",
        )

        body = response.get_json()
        self.assertEqual(200, response.status_code)
        self.assertEqual([], body["faces"])
        self.assertEqual(0, body["detectedFaceCount"])
        self.assertEqual(0, body["processedFaceCount"])
        self.assertEqual(0, service.index_calls)

    def test_rejects_file_over_configured_limit(self):
        service = _FakeEmbeddingService(_analysis([]))
        client = _make_app(service, max_bytes=4).test_client()
        response = client.post(
            "/api/internal/v1/face-analysis",
            data={"file": (io.BytesIO(b"12345"), "snap.jpg")},
            headers={"Authorization": "Bearer internal-secret"},
            content_type="multipart/form-data",
        )

        self.assertEqual(413, response.status_code)
        self.assertEqual("file_too_large", response.get_json()["error"]["code"])
        self.assertEqual(0, service.extract_calls)

    def test_internal_response_body_is_never_forwarded_to_access_log(self):
        service = _FakeEmbeddingService(_analysis([
            {"face_index": 0, "bbox": [1, 2, 3, 4], "confidence": 0.9, "embedding": _embedding(0)},
        ]))
        fake_enroll_module = types.ModuleType("app.routes.enroll")
        fake_enroll_module.embedding_service = service
        fake_enroll_module.enroll_bp = Blueprint("fake_enroll", __name__)

        class TestConfig:
            TESTING = True
            SQLALCHEMY_DATABASE_URI = "sqlite://"
            SQLALCHEMY_TRACK_MODIFICATIONS = False
            ACCESS_LOG_ENABLED = True
            ACCESS_LOG_URL = "http://127.0.0.1:1/api/server-logs"
            ACCESS_LOG_SERVICE_NAME = "test"
            AI_INTERNAL_TOKEN = "internal-secret"
            AI_FACE_ANALYSIS_MAX_IMAGE_BYTES = 1024
            AI_FACE_ANALYSIS_MAX_FACES = 10
            AI_FACE_MODEL_VERSION = "insightface-0.7.3"
            ARCFACE_MODEL_NAME = "buffalo_l"
            MATCH_MIN_SIMILARITY = 0.45

        with patch.dict(sys.modules, {"app.routes.enroll": fake_enroll_module}):
            with patch("app.send_access_log") as send_access_log:
                client = create_app(TestConfig).test_client()
                response = client.post(
                    "/api/internal/v1/face-analysis",
                    data={"file": (io.BytesIO(b"image"), "snap.jpg")},
                    headers={"Authorization": "Bearer internal-secret"},
                    content_type="multipart/form-data",
                )

        self.assertEqual(200, response.status_code)
        forwarded = send_access_log.call_args.kwargs
        self.assertEqual("[internal face-analysis body omitted]", forwarded["request_body"])
        self.assertEqual("[internal face-analysis body omitted]", forwarded["response_body"])
        self.assertNotIn("embedding", forwarded["response_body"])

    def test_create_app_stateless_mode_skips_database_initialization(self):
        service = _FakeEmbeddingService(_analysis([]))
        fake_enroll_module = types.ModuleType("app.routes.enroll")
        fake_enroll_module.embedding_service = service
        fake_enroll_module.enroll_bp = Blueprint("fake_stateless_enroll", __name__)

        class TestConfig:
            TESTING = True
            AI_DATABASE_ENABLED = False
            AI_FACE_ANALYSIS_MATCH_STARS = False
            ACCESS_LOG_ENABLED = False
            AI_INTERNAL_TOKEN = "internal-secret"
            AI_FACE_ANALYSIS_MAX_IMAGE_BYTES = 1024
            AI_FACE_ANALYSIS_MAX_FACES = 10
            AI_FACE_MODEL_VERSION = "insightface-0.7.3"
            ARCFACE_MODEL_NAME = "buffalo_l"
            MATCH_MIN_SIMILARITY = 0.45

        with patch.dict(sys.modules, {"app.routes.enroll": fake_enroll_module}):
            with patch("app.db.init_app") as init_app, patch("app.db.create_all") as create_all:
                client = create_app(TestConfig).test_client()

        self.assertEqual(200, client.get("/api/health").status_code)
        init_app.assert_not_called()
        create_all.assert_not_called()

    def test_database_disabled_forces_face_matching_off(self):
        service = _FakeEmbeddingService(_analysis([
            {"face_index": 0, "bbox": [1, 2, 3, 4], "confidence": 0.9, "embedding": _embedding(0)},
        ]))
        app = Flask(__name__)
        app.config.update(
            TESTING=True,
            AI_DATABASE_ENABLED=False,
            AI_FACE_ANALYSIS_MATCH_STARS=True,
            AI_INTERNAL_TOKEN="internal-secret",
            AI_FACE_ANALYSIS_MAX_IMAGE_BYTES=1024,
            AI_FACE_ANALYSIS_MAX_FACES=10,
            AI_FACE_MODEL_VERSION="insightface-0.7.3",
            ARCFACE_MODEL_NAME="buffalo_l",
            MATCH_MIN_SIMILARITY=0.45,
        )
        app.extensions["embedding_service"] = service
        app.register_blueprint(face_analysis_bp)

        response = app.test_client().post(
            "/api/internal/v1/face-analysis",
            data={"file": (io.BytesIO(b"image"), "face.jpg")},
            headers={"Authorization": "Bearer internal-secret"},
            content_type="multipart/form-data",
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(0, service.index_calls)
        self.assertEqual(0, service.match_calls)


if __name__ == "__main__":
    unittest.main()
