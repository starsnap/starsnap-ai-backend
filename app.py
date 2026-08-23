"""
Flask 애플리케이션 메인 엔트리포인트
클린 아키텍처 패턴으로 구조화됨
"""
from app import create_app

# Flask 애플리케이션 생성
app = create_app()

if __name__ == "__main__":
    # debug/reloader는 설정값을 따르도록 하여 컨테이너에서 불필요한 이중 로딩을 방지한다.
    debug_mode = bool(app.config.get("DEBUG", False))
    app.run(host="0.0.0.0", port=8000, debug=debug_mode, use_reloader=debug_mode)
