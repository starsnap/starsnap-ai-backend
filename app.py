"""
Flask 애플리케이션 메인 엔트리포인트
클린 아키텍처 패턴으로 구조화됨
"""
from app import create_app

# Flask 애플리케이션 생성
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
