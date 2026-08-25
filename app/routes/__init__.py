"""
라우트/API 엔드포인트 모듈
"""
__all__ = ["enroll_bp", "face_analysis_bp"]


def __getattr__(name):
    """무거운 InsightFace 초기화가 단순 route module import에서 실행되지 않게 한다."""
    if name == "enroll_bp":
        from .enroll import enroll_bp

        return enroll_bp
    if name == "face_analysis_bp":
        from .face_analysis import face_analysis_bp

        return face_analysis_bp
    raise AttributeError(name)

