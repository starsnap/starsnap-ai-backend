from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np


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

from app.services.embedding_service import EmbeddingService  # noqa: E402
from config import _require_similarity_env  # noqa: E402


def _vector(index: int, magnitude: float = 1.0) -> np.ndarray:
    value = np.zeros(512, dtype=np.float32)
    value[index] = magnitude
    return value


class _FakeFaceApp:
    def __init__(self, faces):
        self.faces = faces

    def get(self, _image):
        return list(self.faces)


def _service_with_faces(faces) -> EmbeddingService:
    service = EmbeddingService.__new__(EmbeddingService)
    service.face_app = _FakeFaceApp(faces)
    service.max_image_dim = 1280
    service.max_faces = 10
    service.default_min_similarity = 0.45
    service.providers = ["CPUExecutionProvider"]
    return service


class EmbeddingServiceMultiFaceTest(unittest.TestCase):
    def test_similarity_config_rejects_non_finite_or_out_of_range_values(self):
        for invalid_value in ("nan", "inf", "1.01", "-1.01"):
            with self.subTest(invalid_value=invalid_value):
                with patch.dict(os.environ, {"TEST_SIMILARITY": invalid_value}):
                    with self.assertRaises(RuntimeError):
                        _require_similarity_env("TEST_SIMILARITY")

        with patch.dict(os.environ, {"TEST_SIMILARITY": "-1"}):
            self.assertEqual(-1.0, _require_similarity_env("TEST_SIMILARITY"))
        with patch.dict(os.environ, {"TEST_SIMILARITY": "1"}):
            self.assertEqual(1.0, _require_similarity_env("TEST_SIMILARITY"))

    @patch("app.services.embedding_service.bgr_to_rgb", side_effect=lambda image: image)
    @patch("app.services.embedding_service.resize_image_to_max_dim")
    @patch("app.services.embedding_service.decode_image_bytes")
    def test_extracts_largest_faces_in_deterministic_order_and_normalizes(
        self,
        decode_image_bytes,
        resize_image_to_max_dim,
        _bgr_to_rgb,
    ):
        original = np.zeros((200, 400, 3), dtype=np.uint8)
        processed = np.zeros((100, 200, 3), dtype=np.uint8)
        decode_image_bytes.return_value = original
        resize_image_to_max_dim.return_value = (processed, 0.5)
        faces = [
            SimpleNamespace(bbox=np.array([50, 10, 90, 50]), embedding=_vector(2, 3), det_score=0.91),
            SimpleNamespace(bbox=np.array([20, 10, 100, 90]), embedding=_vector(0, 4), det_score=0.99),
            SimpleNamespace(bbox=np.array([10, 10, 50, 50]), embedding=_vector(1, 2), det_score=0.92),
        ]
        service = _service_with_faces(faces)

        result = service.extract_face_embeddings(b"image", max_faces=2)

        self.assertEqual(3, result["detected_face_count"])
        self.assertEqual(2, result["processed_face_count"])
        self.assertTrue(result["truncated"])
        self.assertEqual([0, 1], [face["face_index"] for face in result["faces"]])
        self.assertEqual([40, 20, 160, 160], result["faces"][0]["bbox"])
        self.assertEqual([20, 20, 80, 80], result["faces"][1]["bbox"])
        for face in result["faces"]:
            self.assertEqual((512,), face["embedding"].shape)
            self.assertAlmostEqual(1.0, float(np.linalg.norm(face["embedding"])), places=6)

    @patch("app.services.embedding_service.bgr_to_rgb", side_effect=lambda image: image)
    @patch("app.services.embedding_service.resize_image_to_max_dim")
    @patch("app.services.embedding_service.decode_image_bytes")
    def test_largest_face_wrapper_remains_compatible(
        self,
        decode_image_bytes,
        resize_image_to_max_dim,
        _bgr_to_rgb,
    ):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        decode_image_bytes.return_value = image
        resize_image_to_max_dim.return_value = (image, 1.0)
        service = _service_with_faces([
            SimpleNamespace(bbox=np.array([1, 1, 11, 11]), embedding=_vector(0), det_score=0.8),
            SimpleNamespace(bbox=np.array([5, 4, 35, 24]), embedding=_vector(3, 9), det_score=0.9),
        ])

        result = service.extract_face_embedding(b"image")

        self.assertEqual([5, 4, 30, 20], result["bbox"])
        self.assertEqual(100, result["width"])
        self.assertEqual(100, result["height"])
        self.assertAlmostEqual(1.0, float(np.linalg.norm(result["embedding"])), places=6)

    @patch("app.services.embedding_service.decode_image_bytes", return_value=None)
    def test_invalid_image_is_distinct_from_no_face(self, _decode_image_bytes):
        service = _service_with_faces([])
        self.assertIsNone(service.extract_face_embeddings(b"not-an-image"))

    @patch("app.services.embedding_service.bgr_to_rgb", side_effect=lambda image: image)
    @patch("app.services.embedding_service.resize_image_to_max_dim")
    @patch("app.services.embedding_service.decode_image_bytes")
    def test_rejects_a_selected_face_with_an_empty_bbox_instead_of_silently_succeeding(
        self,
        decode_image_bytes,
        resize_image_to_max_dim,
        _bgr_to_rgb,
    ):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        decode_image_bytes.return_value = image
        resize_image_to_max_dim.return_value = (image, 1.0)
        service = _service_with_faces([
            SimpleNamespace(bbox=np.array([-10, -5, 120, 150]), embedding=_vector(0), det_score=0.9),
            SimpleNamespace(bbox=np.array([120, 120, 150, 160]), embedding=_vector(1), det_score=0.8),
        ])

        with self.assertRaises(ValueError):
            service.extract_face_embeddings(b"image", max_faces=10)

    def test_matches_all_faces_with_one_matrix_and_applies_threshold(self):
        service = _service_with_faces([])
        star_index = {
            "star_ids": ["star-a", "star-b"],
            "embeddings": np.vstack([_vector(0), _vector(1)]),
        }
        matches = service.find_best_star_matches(
            [_vector(0), -_vector(1)],
            star_index=star_index,
            min_similarity=0.45,
        )

        self.assertEqual("star-a", matches[0]["starId"])
        self.assertAlmostEqual(1.0, matches[0]["similarity"], places=6)
        self.assertIsNone(matches[1])

    def test_star_index_query_filters_active_enrolled_stars(self):
        service = _service_with_faces([])
        query = MagicMock()
        query.filter.return_value = query
        query.order_by.return_value = query
        query.all.return_value = [("star-a", _vector(0, 4))]

        with patch("app.services.embedding_service.db") as database:
            database.session.query.return_value = query
            index = service.load_star_embedding_index()

        filter_expressions = query.filter.call_args.args
        self.assertEqual(2, len(filter_expressions))
        self.assertIn("state", str(filter_expressions[0]).lower())
        self.assertIn("true", str(filter_expressions[0]).lower())
        self.assertEqual(["star-a"], index["star_ids"])
        self.assertAlmostEqual(1.0, float(np.linalg.norm(index["embeddings"][0])), places=6)


if __name__ == "__main__":
    unittest.main()
