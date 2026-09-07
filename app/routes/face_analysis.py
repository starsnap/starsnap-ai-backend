"""메인 백엔드가 호출하는 내부 다중 얼굴 분석 API."""
from __future__ import annotations

import hmac
import logging
import uuid
from typing import Any

from flask import Blueprint, current_app, g, jsonify, request


face_analysis_bp = Blueprint(
    "face_analysis",
    __name__,
    url_prefix="/api/internal/v1",
)

logger = logging.getLogger(__name__)


def _get_request_id() -> str:
    supplied = (request.headers.get("X-Request-Id") or "").strip()
    return supplied[:128] if supplied else str(uuid.uuid4())


def _error_response(status: int, code: str, message: str):
    response = jsonify({
        "error": {"code": code, "message": message},
        "requestId": g.face_analysis_request_id,
    })
    response.status_code = status
    return response


@face_analysis_bp.before_request
def _authenticate_internal_request():
    """별도 공유 토큰을 constant-time 비교하고 로그 본문 수집을 차단한다."""
    g.omit_access_log_bodies = True
    g.face_analysis_request_id = _get_request_id()

    authorization = request.headers.get("Authorization") or ""
    scheme, separator, supplied_token = authorization.partition(" ")
    valid_format = bool(separator) and scheme.lower() == "bearer" and bool(supplied_token)
    expected_token = str(current_app.config["AI_INTERNAL_TOKEN"])
    token_matches = hmac.compare_digest(
        (supplied_token if valid_format else "").encode("utf-8"),
        expected_token.encode("utf-8"),
    )
    if not valid_format or not token_matches:
        response = _error_response(
            401,
            "unauthorized",
            "Valid internal bearer token required",
        )
        response.headers["WWW-Authenticate"] = "Bearer"
        return response
    return None


@face_analysis_bp.after_request
def _attach_request_id(response):
    response.headers["X-Request-Id"] = (
        g.get("face_analysis_request_id") or _get_request_id()
    )
    return response


@face_analysis_bp.route("/face-analysis", methods=["POST"])
def analyze_faces():
    """사진의 얼굴 벡터와 선택적으로 스타 매칭 결과를 반환한다."""
    upload = request.files.get("file")
    if upload is None:
        return _error_response(400, "file_required", "Multipart file field is required")
    if not upload.filename:
        return _error_response(400, "filename_empty", "Filename must not be empty")

    max_image_bytes = int(current_app.config["AI_FACE_ANALYSIS_MAX_IMAGE_BYTES"])
    content = upload.stream.read(max_image_bytes + 1)
    if not content:
        return _error_response(400, "file_empty", "Uploaded file must not be empty")
    if len(content) > max_image_bytes:
        return _error_response(
            413,
            "file_too_large",
            f"Uploaded file exceeds {max_image_bytes} bytes",
        )

    configured_max_faces = int(current_app.config["AI_FACE_ANALYSIS_MAX_FACES"])
    max_faces_raw = request.form.get("maxFaces")
    try:
        max_faces = configured_max_faces if max_faces_raw in (None, "") else int(max_faces_raw)
    except ValueError:
        return _error_response(400, "max_faces_invalid", "maxFaces must be an integer")
    if max_faces < 1 or max_faces > configured_max_faces:
        return _error_response(
            400,
            "max_faces_out_of_range",
            f"maxFaces must be between 1 and {configured_max_faces}",
        )

    embedding_service = current_app.extensions.get("embedding_service")
    if embedding_service is None:
        logger.error("[face-analysis] embedding service unavailable request_id=%s", g.face_analysis_request_id)
        return _error_response(503, "service_unavailable", "Face analysis service is unavailable")

    try:
        analysis = embedding_service.extract_face_embeddings(
            content,
            max_faces=max_faces,
        )
        if analysis is None:
            return _error_response(422, "invalid_image", "Uploaded file is not a decodable image")

        extracted_faces = analysis["faces"]
        matches: list[dict[str, Any] | None]
        match_stars = bool(current_app.config.get("AI_DATABASE_ENABLED", True)) and bool(
            current_app.config.get("AI_FACE_ANALYSIS_MATCH_STARS", True)
        )
        if extracted_faces and match_stars:
            # 한 요청의 모든 얼굴이 동일한 등록 Star 스냅샷을 사용한다.
            star_index = embedding_service.load_star_embedding_index()
            matches = embedding_service.find_best_star_matches(
                [face["embedding"] for face in extracted_faces],
                star_index=star_index,
                min_similarity=float(current_app.config["MATCH_MIN_SIMILARITY"]),
            )
        elif extracted_faces:
            # Stateless 모드에서는 운영 DB를 조회하지 않고 임베딩만 반환한다.
            matches = [None] * len(extracted_faces)
        else:
            matches = []

        if len(matches) != len(extracted_faces):
            raise RuntimeError("face match result count does not match extracted face count")

        faces = []
        for face, best_match in zip(extracted_faces, matches):
            embedding = face["embedding"]
            embedding_values = (
                embedding.tolist()
                if hasattr(embedding, "tolist")
                else list(embedding)
            )
            faces.append({
                "faceIndex": int(face["face_index"]),
                "bbox": [int(value) for value in face["bbox"]],
                "detectionConfidence": (
                    float(face["confidence"])
                    if face["confidence"] is not None
                    else None
                ),
                "embedding": [float(value) for value in embedding_values],
                "bestMatch": best_match,
            })

        response = jsonify({
            "schemaVersion": "1",
            "requestId": g.face_analysis_request_id,
            "image": {
                "width": int(analysis["width"]),
                "height": int(analysis["height"]),
            },
            "model": {
                "name": str(current_app.config["ARCFACE_MODEL_NAME"]),
                "version": str(current_app.config["AI_FACE_MODEL_VERSION"]),
                "embeddingDimension": 512,
                "providers": list(embedding_service.active_providers),
            },
            "detectedFaceCount": int(analysis["detected_face_count"]),
            "processedFaceCount": int(analysis["processed_face_count"]),
            "truncated": bool(analysis["truncated"]),
            "faces": faces,
        })
        response.status_code = 200
        return response
    except Exception:  # noqa: BLE001
        logger.exception(
            "[face-analysis] analysis failed request_id=%s",
            g.face_analysis_request_id,
        )
        return _error_response(500, "analysis_failed", "Face analysis failed")
