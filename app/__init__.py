"""
Flask 애플리케이션 팩토리
"""
from flask import Flask
from db import db
from config import Config
from app.models import Star
from app.routes import enroll_bp
from urllib.parse import urlparse, unquote
import logging
import sys


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
    
    # 블루프린트 등록
    app.register_blueprint(enroll_bp)
    
    # 헬스 체크 엔드포인트
    @app.route('/health', methods=['GET'])
    def health():
        return {'status': 'ok'}, 200
    
    return app

