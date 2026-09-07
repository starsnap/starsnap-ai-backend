"""
Flask 애플리케이션 팩토리
"""
from flask import Flask, g, jsonify, request as flask_request
from flask_wtf.csrf import CSRFError, CSRFProtect, generate_csrf
from db import db
from config import Config
from app.models import Star
from app.utils.access_log_sender import send_access_log
from app.utils.jwt_utils import ACCESS_TOKEN_COOKIE_NAME
from urllib.parse import urlparse, unquote
from datetime import datetime, timezone
import logging
import sys
import time


csrf = CSRFProtect()


def _configure_app_logger(app: Flask) -> None:
    """컨테이너/터미널에서 INFO 로그가 보이도록 stdout 핸들러를 고정한다."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    app.logger.handlers.clear()
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    app.logger.propagate = False


def _log_db_target(app: Flask) -> None:
    """DB 접속 대상(host/port/db)만 출력하고 민감정보는 노출하지 않는다."""
    db_uri = app.config.get("SQLALCHEMY_DATABASE_URI", "")
    if not db_uri:
        app.logger.info("[DB] SQLALCHEMY_DATABASE_URI is empty")
        return
    else:
        app.logger.info("[DB] SQLALCHEMY_DATABASE_URI is not empty")

    try:
        parsed = urlparse(db_uri)
        host = parsed.hostname or "(none)"
        port = parsed.port or "(default)"
        database = unquote((parsed.path or "").lstrip("/")) or "(none)"
        scheme = parsed.scheme or "(unknown)"
        app.logger.info("[DB] target: scheme=%s, host=%s, port=%s, db=%s", scheme, host, port, database)
    except Exception as e:
        app.logger.warning("[DB] could not parse SQLALCHEMY_DATABASE_URI: %s", e)


def create_app(config_class=Config):
    """
    Flask 애플리케이션 생성 및 초기화
    
    Args:
        config_class: 설정 클래스
        
    Returns:
        Flask: 초기화된 Flask 애플리케이션
    """
    app = Flask(__name__)
    
    _configure_app_logger(app)

    # 설정 로드
    app.config.from_object(config_class)
    # 전역 자동 검사는 Bearer/진단 POST까지 막으므로 아래 쿠키 경계에서만 호출한다.
    app.config["WTF_CSRF_CHECK_DEFAULT"] = False

    # 모델 초기화는 애플리케이션 생성 시점까지 지연해 route 단위 테스트에서
    # 실제 InsightFace 모델을 올리지 않고 fake 서비스를 주입할 수 있게 한다.
    from app.routes.enroll import embedding_service, enroll_bp
    from app.routes.face_analysis import face_analysis_bp

    app.extensions["embedding_service"] = embedding_service

    if database_enabled:
        # 실제 DB 연결 대상을 시작 시 1회 로그로 출력
        _log_db_target(app)

        # 데이터베이스 초기화
        db.init_app(app)

        # 애플리케이션 컨텍스트에서 테이블 생성 (DB 연결이 가능한 경우에만)
        try:
            with app.app_context():
                db.create_all()
        except Exception as e:
            app.logger.warning("Warning: Could not create database tables: %s", e)
            app.logger.warning("Make sure your database is running and configured correctly.")
    else:
        app.logger.info(
            "AI database disabled; running stateless face-analysis mode"
        )
    
    # 블루프린트 등록
    app.register_blueprint(enroll_bp)
    app.register_blueprint(face_analysis_bp)

    @app.errorhandler(CSRFError)
    def _handle_csrf_error(_error):
        return jsonify({"error": "CSRF token missing or invalid"}), 400

    @app.get("/api/csrf-token")
    def csrf_token():
        response = jsonify({"csrfToken": generate_csrf()})
        response.headers["Cache-Control"] = "no-store"
        return response

    # -----------------------------------------------------------------------
    # Access Log 전송 훅
    # -----------------------------------------------------------------------
    @app.before_request
    def _before_access_log():
        """요청 시작 시각을 기록하고 요청 바디를 미리 캐시한다."""
        g.access_log_start = time.monotonic()
        g.access_log_requested_at = datetime.now(timezone.utc)
        # multipart/form-data 는 파일 본문을 읽지 않는다.
        if not (flask_request.content_type or "").lower().startswith("multipart/form-data"):
            # get_data()를 여기서 한 번 호출해 두면 Flask 내부에 캐시되어
            # 이후 뷰 함수에서도 정상적으로 읽을 수 있다.
            flask_request.get_data()

    csrf.init_app(app)

    @app.before_request
    def _protect_cookie_authenticated_write():
        """브라우저 쿠키가 권한을 부여하는 상태 변경 요청만 CSRF 검증한다."""
        if (
            flask_request.method in {"POST", "PUT", "PATCH", "DELETE"}
            and flask_request.cookies.get(ACCESS_TOKEN_COOKIE_NAME)
        ):
            csrf.protect()

    @app.after_request
    def _after_access_log(response):
        """요청이 끝난 뒤 처리 결과를 외부 로그 서비스로 비동기 전송한다."""
        if not app.config.get("ACCESS_LOG_ENABLED", True):
            return response
        try:
            # 밀리초 단위 소수점을 유지하여 0ms로 표시되는 빈도를 줄입니다.
            # 소수점은 3자리까지 반올림하여 JSON으로 전송합니다 (예: 0.452 ms).
            now_ts = time.monotonic()
            delta = now_ts - g.access_log_start
            elapsed_ms = round(delta * 1000, 3)

            # 요청 정보 수집
            req_headers = "\n".join(
                f"{k}: {v}" for k, v in flask_request.headers.items()
            )
            omit_bodies = bool(getattr(g, "omit_access_log_bodies", False))
            if omit_bodies:
                req_body = "[internal face-analysis body omitted]"
            elif (flask_request.content_type or "").lower().startswith("multipart/form-data"):
                req_body = "\n".join(
                    f"{k}={v}" for k, v in flask_request.form.items()
                ) or "[multipart/form-data omitted]"
            else:
                req_body = flask_request.get_data(as_text=True) or ""

            # 응답 정보 수집 (바이너리 응답은 빈 문자열로 처리)
            content_type = response.content_type or ""
            if omit_bodies:
                resp_body = "[internal face-analysis body omitted]"
            elif "text" in content_type or "json" in content_type or "xml" in content_type:
                resp_body = response.get_data(as_text=True) or ""
            else:
                resp_body = ""

            resp_headers = "\n".join(
                f"{k}: {v}" for k, v in response.headers.items()
            )

            send_access_log(
                url=app.config.get("ACCESS_LOG_URL"),
                secret=app.config.get("ACCESS_LOG_SECRET", ""),
                service_name=app.config.get("ACCESS_LOG_SERVICE_NAME", "starsnap-ai-backend"),
                path=flask_request.path,
                method=flask_request.method,
                status_code=response.status_code,
                ip_address=flask_request.remote_addr,
                response_time_ms=elapsed_ms,
                requested_at=g.access_log_requested_at,
                user_agent=flask_request.headers.get("User-Agent"),
                request_headers=req_headers,
                request_body=req_body,
                response_headers=resp_headers,
                response_body=resp_body,
                query_params=flask_request.query_string.decode("utf-8", errors="replace"),
            )
        except Exception:  # noqa: BLE001
            app.logger.exception("[access-log] Error collecting request data for log forwarding")
        return response

    # 헬스 체크 엔드포인트
    @app.route('/api/health', methods=['GET'])
    def health():
        return {'status': 'ok'}, 200
    
    return app

