"""
등록(Enroll) 관련 라우트
"""
from flask import Blueprint, current_app, request, jsonify, send_file, g
import io
import logging
from app.utils.http_forward import forward_multipart_request, build_multipart_payload, forward_request
import json
from app.services import EmbeddingService
from app.utils.jwt_utils import ACCESS_TOKEN_COOKIE_NAME, require_jwt, require_admin
from config import Config
from app.utils.access_log_sender import send_access_log
from datetime import datetime, timezone
import time
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

# 블루프린트 생성
enroll_bp = Blueprint('enroll', __name__, url_prefix='/api')

# presign / multipart 네트워크 요청을 백그라운드에서 병렬 실행하기 위한 스레드풀
_enroll_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="enroll")

# 임베딩 서비스 인스턴스 (전역)
embedding_service = EmbeddingService(
    providers=Config.ARCFACE_PROVIDERS,
    model_name=Config.ARCFACE_MODEL_NAME,
    det_size=(Config.ARCFACE_DET_SIZE, Config.ARCFACE_DET_SIZE),
    default_min_similarity=Config.MATCH_MIN_SIMILARITY,
)


def _read_bounded_upload(upload):
    """Read an authenticated diagnostic upload without unbounded memory use."""
    max_bytes = int(Config.AI_FACE_ANALYSIS_MAX_IMAGE_BYTES)
    content = upload.stream.read(max_bytes + 1)
    if not content:
        return None, (jsonify({"error": "file is empty"}), 400)
    if len(content) > max_bytes:
        return None, (jsonify({"error": f"file exceeds {max_bytes} bytes"}), 413)
    return content, None


def _uses_presigned_upload_api(url: str) -> bool:
    """Whether PHOTO_API_URL targets the main backend presign endpoint."""
    try:
        parsed = urlparse(url)
    except (TypeError, ValueError):
        return False

    normalized_path = "/" + parsed.path.strip("/")
    return bool(parsed.scheme and parsed.netloc) and normalized_path.endswith("/api/file/photo")


def _database_disabled_response():
    """Return a stable response for legacy DB-backed routes in stateless mode."""
    if current_app.config.get("AI_DATABASE_ENABLED", True):
        return None
    return jsonify({
        "error": "database-backed API is disabled",
        "code": "database_disabled",
    }), 503


@enroll_bp.route("/enroll", methods=["POST"])
@require_jwt
@require_admin
def enroll():
    """
    이미지 업로드 후 star.face_image_vector에 임베딩 저장
    
    핵심 최적화: 임베딩 추출(AI)과 presign POST(네트워크) 요청을 병렬로 실행한다.
    두 작업이 독립적이므로 총 소요 시간 ≈ max(임베딩, presign) + PUT 업로드.

    Request:
        - file: 이미지 파일 (optional, presign-only 흐름에서는 생략 가능)
        - star_id: 스타 ID (required)
        
    Returns:
        - status: ok
        - star_id: 스타 ID
        - embedding_dim: 임베딩 차원
    """
    disabled_response = _database_disabled_response()
    if disabled_response is not None:
        return disabled_response

    logger = logging.getLogger(__name__)
    t_total = time.monotonic()
    t_parse = t_total

    # ── 1. 입력 파싱 ──────────────────────────────────────────────────────────
    file = request.files.get('file')
    star_id = request.form.get('star_id')
    if not star_id:
        return jsonify({"error": "star_id required"}), 400

    if file is not None:
        if file.filename == '':
            return jsonify({"error": "filename is empty"}), 400
        original_filename = file.filename or "upload.bin"
        original_content_type = file.mimetype
        content, upload_error = _read_bounded_upload(file)
        if upload_error is not None:
            return upload_error
    else:
        original_filename = "upload.bin"
        original_content_type = None
        content = b""

    metadata_fields = {}
    for key in ("aiState", "dateTaken", "source"):
        value = request.form.get(key)
        if value is not None:
            if key == "aiState":
                v = str(value).strip().lower()
                if v in {"1", "true", "yes", "on"}:
                    metadata_fields[key] = True
                elif v in {"0", "false", "no", "off"}:
                    metadata_fields[key] = False
                else:
                    metadata_fields[key] = value
            else:
                metadata_fields[key] = value
    if file is not None:
        metadata_fields["contentType"] = original_content_type
        metadata_fields["fileSize"] = len(content)

    access_token = getattr(g, "access_token", "")
    forwarded_cookie = (
        f"{ACCESS_TOKEN_COOKIE_NAME}={access_token}"
        if isinstance(access_token, str) and access_token
        else None
    )
    uses_presigned_upload_api = _uses_presigned_upload_api(Config.PHOTO_API_URL)
    logger.info("[timing] step1 input_parse=%.0fms star_id=%s file_present=%s metadata_keys=%s",
                (time.monotonic() - t_parse) * 1000, star_id, file is not None, list(metadata_fields.keys()))

    # ── 2. presign 요청을 백그라운드 스레드에 제출 (파일이 있을 때만 병렬화 의미 있음)
    #       파일이 없는 경우에도 동일 스레드로 처리하되 메인 스레드가 블로킹됨.
    presign_future = None
    if uses_presigned_upload_api:
        _json_body = json.dumps(metadata_fields or {}).encode("utf-8")
        _hdrs = {"Content-Type": "application/json"}
        if forwarded_cookie:
            _hdrs["Cookie"] = forwarded_cookie

        # 클로저 캡처용 지역 변수 복사
        _presign_url = Config.PHOTO_API_URL
        _presign_timeout = Config.PHOTO_API_TIMEOUT_SECONDS

        def _do_presign():
            return forward_request(
                url=_presign_url,
                method="POST",
                body=_json_body,
                headers=_hdrs,
                timeout=_presign_timeout,
                expect_json=True,
                error_prefix="photo",
            )

        presign_future = _enroll_executor.submit(_do_presign)
        logger.debug("[timing] presign future submitted for star_id=%s", star_id)

    # ── 3. 임베딩 추출 (메인 스레드) — presign 요청과 병렬로 진행됨 ───────────
    info = None
    if file is not None:
        logger.debug("Extracting embedding for star_id=%s filename=%s content_len=%d",
                     star_id, original_filename, len(content))
        t_emb = time.monotonic()
        info = embedding_service.extract_face_embedding(content)
        logger.info("[timing] step2 embedding_extraction=%.0fms star_id=%s",
                    (time.monotonic() - t_emb) * 1000, star_id)

        if info is None:
            logger.info("No face detected in uploaded image for star_id=%s", star_id)
            return jsonify({"error": "no face detected"}), 404

        t_db = time.monotonic()
        updated, save_reason = embedding_service.save_star_embedding_with_reason(
            star_id=star_id,
            embedding=info['embedding'],
        )
        logger.info("[timing] step3 db_save=%.0fms star_id=%s",
                    (time.monotonic() - t_db) * 1000, star_id)

        if not updated:
            logger.warning("Failed to update star.face_image_vector for star_id=%s reason=%s",
                           star_id, save_reason)
            if save_reason == "star_not_found":
                return jsonify({"error": "star not found"}), 404
            return jsonify({"error": "failed to save face_image_vector", "reason": save_reason}), 500

        logger.info("Saved embedding for star_id=%s dim=%s", star_id, int(info['embedding'].shape[0]))
    else:
        logger.warning(
            "Enroll called without file for star_id=%s; content_type=%s files_keys=%s form_keys=%s",
            star_id, request.content_type,
            list(request.files.keys()), list(request.form.keys()),
        )

    # ── 4. presign 결과 수집 (이미 완료됐거나 잠시 대기) ────────────────────
    presign_result = None
    presign_status = None
    presign_error = None
    presign_raw = None
    presign_headers = {}

    if presign_future is not None:
        t_wait = time.monotonic()
        resp_body, resp_status, resp_headers = presign_future.result()
        waited_ms = (time.monotonic() - t_wait) * 1000
        logger.info("[timing] presign wait(future)=%.0fms star_id=%s", waited_ms, star_id)

        presign_raw = resp_body
        presign_headers = resp_headers

        if resp_status is None:
            presign_error = {"error": "no response from upstream"}
            presign_status = 502
        elif resp_status >= 400:
            presign_error = (resp_body if isinstance(resp_body, dict)
                             else {"error": "upstream http error", "upstream_body": resp_body})
            presign_status = resp_status
        else:
            if isinstance(resp_body, dict):
                presign_result = resp_body
            else:
                try:
                    if isinstance(resp_body, (bytes, bytearray)):
                        decoded = resp_body.decode("utf-8")
                    elif resp_body is None:
                        decoded = ""
                    else:
                        decoded = str(resp_body)
                    parsed = json.loads(decoded) if decoded else None
                    if isinstance(parsed, dict):
                        presign_result = parsed
                    else:
                        presign_result = None
                        logger.warning("Presign response parsed to non-dict JSON: %r", parsed)
                except Exception:
                    logger.warning(
                        "Failed to parse presign response as JSON. body_type=%s header_names=%s",
                        type(resp_body).__name__,
                        sorted(resp_headers.keys()),
                    )
                    presign_result = None
            presign_status = resp_status

    # ── 5. 파일 없음 → presign 결과만 반환 ──────────────────────────────────
    if file is None:
        if presign_result is not None:
            return jsonify(presign_result), presign_status
        return jsonify(presign_error or {"error": "no file provided"}), presign_status or 400

    # ── 6. starsnap-backend 흐름: presignedUrl로 PUT 업로드 ──────────────────
    if uses_presigned_upload_api:
        t_photo = time.monotonic()
        if not isinstance(presign_result, dict) or not presign_result.get("presignedUrl"):
            logger.error(
                "presignedUrl missing: status=%s result_keys=%s header_names=%s",
                presign_status,
                sorted(presign_result.keys()) if isinstance(presign_result, dict) else [],
                sorted(presign_headers.keys()),
            )
            if isinstance(presign_result, dict):
                for alt_key in ("presigned_url", "presignedURL", "presignUrl", "uploadUrl", "url"):
                    if presign_result.get(alt_key):
                        logger.warning("Found alternative presign key %s, using it", alt_key)
                        presign_result["presignedUrl"] = presign_result[alt_key]
                        break

            if not (isinstance(presign_result, dict) and presign_result.get("presignedUrl")):
                return jsonify({"error": "presignedUrl missing from upstream response"}), 502

        def _to_header_value(value):
            if isinstance(value, bool):
                return "true" if value else "false"
            return str(value)

        upload_headers = {}
        required_headers = presign_result.get("requiredHeaders")
        if isinstance(required_headers, dict):
            upload_headers.update({str(k): _to_header_value(v) for k, v in required_headers.items()})

        if "aiState" in metadata_fields and "x-amz-meta-ai-state" not in upload_headers:
            upload_headers["x-amz-meta-ai-state"] = _to_header_value(metadata_fields["aiState"])
        if "dateTaken" in metadata_fields and "x-amz-meta-date-taken" not in upload_headers:
            upload_headers["x-amz-meta-date-taken"] = _to_header_value(metadata_fields["dateTaken"])
        if "source" in metadata_fields and "x-amz-meta-source" not in upload_headers:
            upload_headers["x-amz-meta-source"] = _to_header_value(metadata_fields["source"])
        if "x-amz-meta-user-id" not in upload_headers:
            upload_headers["x-amz-meta-user-id"] = _to_header_value(star_id)

        lower_keys = {k.lower() for k in upload_headers.keys()}
        if original_content_type and "content-type" not in lower_keys:
            upload_headers["Content-Type"] = original_content_type

        t_put = time.monotonic()
        upload_body, upload_status, upload_resp_headers = forward_request(
            url=presign_result["presignedUrl"],
            method="PUT",
            body=content,
            headers=upload_headers,
            timeout=Config.PHOTO_API_TIMEOUT_SECONDS,
            expect_json=False,
            error_prefix="presigned upload",
        )
        logger.info("[timing] presigned PUT=%.0fms status=%s star_id=%s",
                    (time.monotonic() - t_put) * 1000, upload_status, star_id)

        if upload_status is None:
            logger.info("[timing] step4 photo_transfer=%.0fms mode=presign+put status=%s star_id=%s",
                        (time.monotonic() - t_photo) * 1000, upload_status, star_id)
            return jsonify({"error": "no response from presigned upload"}), 502
        if upload_status >= 400:
            logger.info("[timing] step4 photo_transfer=%.0fms mode=presign+put status=%s star_id=%s",
                        (time.monotonic() - t_photo) * 1000, upload_status, star_id)
            return jsonify({
                "error": "presigned upload failed",
                "upstream_body": upload_body,
            }), upload_status

        # Access log for presigned upload (비동기 전송)
        try:
            upload_requested_at = datetime.now(timezone.utc)
            _start: float = getattr(g, "access_log_start", 0.0) or 0.0
            # 기록용으로 소수점 밀리초를 유지합니다 (예: 0.452 ms).
            upload_elapsed_ms = round((time.monotonic() - _start) * 1000, 3) if _start else 0.0
        except Exception:
            upload_requested_at = datetime.now(timezone.utc)
            upload_elapsed_ms = 0

        _presigned_full_url = presign_result.get("presignedUrl", "")
        try:
            parsed_url = urlparse(_presigned_full_url)
            # 서버 스키마(path)는 URL path component만 기대하므로 전체 URL 대신 path만 전송한다.
            presign_path = parsed_url.path or "/"
            presign_query = parsed_url.query or ""
        except Exception:
            presign_path = "/"
            presign_query = ""

        try:
            req_headers_str = "\n".join(f"{k}: {v}" for k, v in upload_headers.items())
        except Exception:
            req_headers_str = ""

        req_body_str = f"[binary {len(content)} bytes]" if content else ""

        try:
            resp_headers_str = "\n".join(f"{k}: {v}" for k, v in (upload_resp_headers or {}).items())
        except Exception:
            resp_headers_str = ""

        resp_body_str = ""
        try:
            if isinstance(upload_body, (bytes, bytearray)):
                try:
                    resp_body_str = upload_body.decode("utf-8")
                except Exception:
                    resp_body_str = ""
            elif isinstance(upload_body, dict):
                resp_body_str = json.dumps(upload_body, ensure_ascii=False)
            else:
                resp_body_str = str(upload_body) if upload_body is not None else ""
        except Exception:
            resp_body_str = ""

        try:
            send_access_log(
                url=Config.ACCESS_LOG_URL,
                secret=Config.ACCESS_LOG_SECRET,
                service_name=Config.ACCESS_LOG_SERVICE_NAME,
                path=presign_path,
                method="PUT",
                status_code=upload_status or 502,
                ip_address=request.remote_addr,
                response_time_ms=upload_elapsed_ms,
                requested_at=upload_requested_at,
                user_agent=request.headers.get("User-Agent"),
                request_headers=req_headers_str,
                request_body=req_body_str,
                response_headers=resp_headers_str,
                response_body=resp_body_str,
                query_params=presign_query,
            )
        except Exception:
            logger.exception("Failed to send access log for presigned upload")

        logger.info("[timing] step4 photo_transfer=%.0fms mode=presign+put status=%s star_id=%s",
                    (time.monotonic() - t_photo) * 1000, upload_status, star_id)

    else:
        # ── 7. non-starsnap-backend: 기존 multipart 전송 ─────────────────────
        t_photo = time.monotonic()
        multipart_body, boundary = build_multipart_payload(
            file_bytes=content,
            filename=original_filename,
            file_field_name="file",
            fields=metadata_fields,
            content_type=original_content_type,
        )

        t_mp = time.monotonic()
        forward_error, forward_status = forward_multipart_request(
            url=Config.PHOTO_API_URL,
            multipart_body=multipart_body,
            boundary=boundary,
            headers={"Cookie": forwarded_cookie} if forwarded_cookie else None,
            timeout=Config.PHOTO_API_TIMEOUT_SECONDS,
            error_prefix="photo",
        )
        logger.info("[timing] multipart forward=%.0fms star_id=%s",
                    (time.monotonic() - t_mp) * 1000, star_id)

        if forward_error is not None:
            logger.info("[timing] step4 photo_transfer=%.0fms mode=multipart status=%s star_id=%s",
                        (time.monotonic() - t_photo) * 1000, forward_status, star_id)
            return jsonify(forward_error), forward_status

        logger.info("[timing] step4 photo_transfer=%.0fms mode=multipart status=%s star_id=%s",
                    (time.monotonic() - t_photo) * 1000, forward_status, star_id)

    logger.info("[timing] enroll TOTAL=%.0fms star_id=%s",
                (time.monotonic() - t_total) * 1000, star_id)
    return jsonify({
        "status": "ok",
        "star_id": star_id,
        "embedding_dim": int(info['embedding'].shape[0])
    }), 201


@enroll_bp.route("/embedding/star/<string:star_id>", methods=["GET"])
@require_jwt
@require_admin
def get_embedding(star_id):
    """
    Star ID로 임베딩 벡터 조회 (디버그용)

    Args:
        star_id (str): 스타 ID

    Returns:
        - star_id: 스타 ID
        - embedding_dim: 임베딩 차원
        - embedding_preview: 임베딩의 첫 10개 값
    """
    disabled_response = _database_disabled_response()
    if disabled_response is not None:
        return disabled_response

    vec = embedding_service.get_star_embedding_vector(star_id)

    if vec is None:
        return jsonify({"error": "not found"}), 404

    return jsonify({
        "star_id": star_id,
        "embedding_dim": int(vec.shape[0]),
        "embedding_preview": vec[:10].tolist()
    }), 200
@enroll_bp.route("/match/star", methods=["POST"])
@require_jwt
@require_admin
def match_star():
    """업로드한 얼굴과 가장 유사한 Star 정보를 반환한다.

    임계값 미만이어도 비교 가능한 Star가 있으면 최고 유사도 결과를 반환한다.
    """
    disabled_response = _database_disabled_response()
    if disabled_response is not None:
        return disabled_response

    if 'file' not in request.files:
        return jsonify({"error": "file required"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "filename is empty"}), 400

    content, upload_error = _read_bounded_upload(file)
    if upload_error is not None:
        return upload_error

    info = embedding_service.extract_face_embedding(content)
    if info is None:
        return jsonify({"error": "no face detected"}), 404

    min_similarity = Config.MATCH_MIN_SIMILARITY
    result = embedding_service.find_most_similar_star(
        info['embedding'],
        min_similarity=-1.0,
    )
    if result is None:
        return jsonify({"error": "no enrolled star embeddings to compare"}), 404

    similarity = float(result['similarity'])
    threshold_passed = similarity >= min_similarity

    return jsonify({
        "status": "ok",
        "threshold": {
            "min_similarity": float(min_similarity),
            "passed": threshold_passed,
        },
        "query": {
            "embedding_dim": int(info['embedding'].shape[0]),
            "bbox": info['bbox'],
            "confidence": info['confidence']
        },
        "match": {
            "star": result['star'],
            "similarity": similarity
        }
    }), 200
@enroll_bp.route("/test/largest-face", methods=["POST"])
@require_jwt
@require_admin
def test_largest_face():
    """업로드 이미지에서 가장 큰 얼굴 1개를 잘라 파일로 반환한다."""
    if 'file' not in request.files:
        return jsonify({"error": "file required"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "filename is empty"}), 400

    content, upload_error = _read_bounded_upload(file)
    if upload_error is not None:
        return upload_error

    result = embedding_service.extract_largest_face_for_test(content)
    if result is None:
        return jsonify({"error": "no face detected"}), 404

    response = send_file(
        io.BytesIO(result["face_image_bytes"]),
        mimetype="image/jpeg",
        as_attachment=True,
        download_name="largest-face.jpg",
    )
    response.headers["X-Face-Bbox"] = ",".join(map(str, result["bbox"]))
    response.headers["X-Face-Confidence"] = str(result["confidence"])
    response.headers["X-Source-Width"] = str(result["width"])
    response.headers["X-Source-Height"] = str(result["height"])
    return response


@enroll_bp.route("/test/face-vector", methods=["POST"])
@require_jwt
@require_admin
def test_face_vector():
    """업로드한 인물 사진에서 얼굴 임베딩 벡터를 추출해 JSON으로 반환한다.

    Request:
        - file: 이미지 파일 (required)

    Returns:
        - embedding: 512차원 얼굴 벡터 전체
        - embedding_dim: 벡터 차원
        - bbox: 검출된 얼굴 좌표
        - confidence: 검출 신뢰도
        - width/height: 원본 이미지 크기
    """
    if 'file' not in request.files:
        return jsonify({"error": "file required"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "filename is empty"}), 400

    content, upload_error = _read_bounded_upload(file)
    if upload_error is not None:
        return upload_error

    max_dim_raw = request.form.get("max_dim")
    try:
        max_dim = int(max_dim_raw) if max_dim_raw not in (None, "") else Config.ARCFACE_MAX_IMAGE_DIM
    except ValueError:
        return jsonify({"error": "max_dim must be an integer"}), 400

    info = embedding_service.extract_face_embedding(content, max_dim=max_dim)
    if info is None:
        return jsonify({"error": "no face detected"}), 404

    embedding = info["embedding"]
    embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    return jsonify({
        "status": "ok",
        "device": embedding_service.device,
        "active_providers": embedding_service.active_providers,
        "embedding_dim": int(len(embedding_list)),
        "embedding": embedding_list,
        "bbox": info["bbox"],
        "confidence": info["confidence"],
        "width": info["width"],
        "height": info["height"],
        "max_dim": int(max_dim) if max_dim is not None else None,
    }), 200
