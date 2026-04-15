"""
JWT 검증 유틸리티
starsnap-backend(Spring)에서 발급한 Access Token을 검증한다.

토큰 구조 (Spring JwtProvider 기준):
  - 알고리즘 : HS256
  - 헤더 JWT : "access"
  - Payload
      jti       : 사용자 UUID
      authority : 권한 (USER / ADMIN / STAR)
"""
from functools import wraps

import jwt
from flask import request, jsonify, g

from config import Config


def _decode_access_token(token: str) -> dict:
    """
    Access Token 디코드 & 검증.

    Returns:
        payload dict

    Raises:
        jwt.ExpiredSignatureError: 토큰 만료
        jwt.InvalidTokenError   : 서명/형식 오류
        ValueError              : JWT 헤더가 "access"가 아닐 때
    """
    # 1) 헤더 타입 확인 (JWT 키만 허용)
    header = jwt.get_unverified_header(token)
    jwt_value = header.get("JWT")
    if jwt_value != "access":
        raise ValueError(f"Invalid token type: JWT={jwt_value}")

    # 2) 서명 + 만료 검증
    payload = jwt.decode(
        token,
        Config.JWT_ACCESS_SECRET,
        algorithms=["HS256"],
        options={"verify_typ": False},  # typ="access"는 비표준이므로 PyJWT 기본 검사 스킵
    )
    return payload


def get_current_user() -> dict | None:
    """
    현재 요청의 인증된 사용자 정보를 반환한다.
    require_jwt 데코레이터가 적용된 뷰 안에서만 유효하다.

    Returns:
        {"user_id": str, "authority": str} 또는 None
    """
    user_id = getattr(g, "user_id", None)
    authority = getattr(g, "authority", None)
    if user_id is None:
        return None
    return {"user_id": user_id, "authority": authority}


def require_jwt(f):
    """
    JWT Access Token 검증 데코레이터.

    Authorization: Bearer <token> 헤더를 검증하고,
    성공 시 g.user_id / g.authority 에 사용자 정보를 저장한다.

    Usage:
        @enroll_bp.route("/some-protected", methods=["POST"])
        @require_jwt
        def some_protected_view():
            user = get_current_user()
            ...
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Authorization header missing or invalid"}), 401

        token = auth_header[len("Bearer "):]

        try:
            payload = _decode_access_token(token)
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except ValueError as e:
            return jsonify({"error": str(e)}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        # Flask g 에 사용자 정보 저장
        g.user_id = payload.get("jti")        # Spring에서 setId(uuid)
        g.authority = payload.get("authority") # Spring에서 claim(AUTHORITY, authority)

        return f(*args, **kwargs)

    return decorated


# jwt ADMIN 권한 허용
def require_admin(f):
    """JWT 인증 이후 ADMIN 권한만 허용하는 데코레이터."""
    @wraps(f)
    def decorated(*args, **kwargs):
        authority = getattr(g, "authority", None)
        if authority != "ADMIN":
            return jsonify({"error": "Admin only"}), 403
        return f(*args, **kwargs)

    return decorated

# jwt ADMIN OR USER 권한 접근 허용
def require_user_or_admin(f):
    """JWT 인증 이후 USER 또는 ADMIN 권한만 허용하는 데코레이터."""
    @wraps(f)
    def decorated(*args, **kwargs):
        authority = getattr(g, "authority", None)
        if authority not in {"USER", "ADMIN"}:
            return jsonify({"error": "User or Admin only"}), 403
        return f(*args, **kwargs)

    return decorated


