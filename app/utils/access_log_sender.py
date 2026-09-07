"""
Access Log Sender
모든 요청의 처리 결과를 외부 로그 수집 서비스(POST /api/server-logs)로
백그라운드 ThreadPool을 통해 비동기 전송합니다.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

logger = logging.getLogger(__name__)

# 백그라운드 전송용 스레드풀 (앱 전체에서 단 1개)
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="access-log")

_IPV4_RE = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
_SENSITIVE_HEADER_NAMES = {
    "authorization",
    "cookie",
    "set-cookie",
    "proxy-authorization",
    "x-api-key",
}
_SENSITIVE_FIELD_NAMES = {
    "password",
    "passcode",
    "accesstoken",
    "refreshtoken",
    "registertoken",
    "token",
    "authorization",
    "cookie",
    "secret",
    "secretkey",
    "credential",
    "verificationcode",
    "verifycode",
    "xamzcredential",
    "xamzsignature",
    "xamzsecuritytoken",
    "apikey",
    "session",
    "sessionid",
    "otp",
}
_FORM_SECRET_RE = re.compile(
    r"(?i)((?:^|[&\s])(?:password|passcode|access[_-]?token|refresh[_-]?token|"
    r"register[_-]?token|token|authorization|cookie|secret|credential|"
    r"verification[_-]?code|verify[_-]?code|x-amz-credential|x-amz-signature|"
    r"x-amz-security-token)=)[^&\s]*"
)
_JSON_SECRET_RE = re.compile(
    r"""(?i)(["']?(?:password|passcode|access[_-]?token|refresh[_-]?token|"""
    r"""register[_-]?token|token|authorization|cookie|secret|credential|"""
    r"""verification[_-]?code|verify[_-]?code|x-amz-credential|x-amz-signature|"""
    r"""x-amz-security-token)["']?\s*:\s*)(["'][^"']*["']|[^,}\s]+)"""
)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _is_ipv4(addr: str) -> bool:
    return bool(_IPV4_RE.match(addr or ''))


def _sanitize_ip(addr: Optional[str]) -> str:
    """IPv4 포맷이 아니면 0.0.0.0 을 반환한다."""
    if addr and _is_ipv4(addr):
        return addr
    return "0.0.0.0"


def _truncate(s: str, limit: int = 10_000) -> str:
    """10,000자를 초과하면 앞에서 잘라 ...(truncated) 를 붙인다."""
    if s and len(s) > limit:
        return s[:limit] + "...(truncated)"
    return s or ""


def _fmt_dt(dt: datetime) -> str:
    """datetime → ISO-8601 Z 포맷 (예: 2026-05-11T11:20:35.120Z)"""
    ms = dt.microsecond // 1000
    return dt.strftime('%Y-%m-%dT%H:%M:%S') + f'.{ms:03d}Z'


def _is_sensitive_field(name: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return (
        normalized in _SENSITIVE_FIELD_NAMES
        or normalized.endswith("password")
        or normalized.endswith("token")
        or normalized.endswith("secret")
        or normalized.endswith("credential")
    )


def redact_query_params(raw: str) -> str:
    if not raw:
        return ""
    try:
        values = parse_qsl(raw, keep_blank_values=True)
        return urlencode(
            [(key, "[REDACTED]" if _is_sensitive_field(key) else value) for key, value in values],
            doseq=True,
        )
    except Exception:
        return _FORM_SECRET_RE.sub(lambda match: match.group(1) + "[REDACTED]", raw)


def _redact_value(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _is_sensitive_field(key) else _redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = urlsplit(value)
            if parsed.scheme and parsed.netloc and parsed.query:
                return urlunsplit((
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    redact_query_params(parsed.query),
                    parsed.fragment,
                ))
        except Exception:
            pass
    return value


def redact_body(raw: str) -> str:
    if not raw:
        return ""
    try:
        return json.dumps(_redact_value(json.loads(raw)), ensure_ascii=False)
    except Exception:
        json_redacted = _JSON_SECRET_RE.sub(
            lambda match: match.group(1) + '"[REDACTED]"',
            raw,
        )
        return _FORM_SECRET_RE.sub(
            lambda match: match.group(1) + "[REDACTED]",
            json_redacted,
        )


def redact_headers(raw: str) -> str:
    redacted_lines = []
    for line in (raw or "").splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip().lower() in _SENSITIVE_HEADER_NAMES:
            redacted_lines.append(f"{name}: [REDACTED]")
        else:
            redacted_lines.append(line)
    return "\n".join(redacted_lines)


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def send_access_log(
    *,
    url: str,
    secret: str,
    service_name: str,
    path: str,
    method: str,
    status_code: int,
    ip_address: Optional[str],
    response_time_ms: float,
    requested_at: datetime,
    user_agent: Optional[str],
    request_headers: str,
    request_body: str,
    response_headers: str,
    response_body: str,
    query_params: str,
) -> None:
    """백그라운드 스레드에서 액세스 로그를 전송한다. 절대 예외를 올리지 않는다."""
    if not url or not secret:
        return
    _executor.submit(
        _do_send,
        url=url,
        secret=secret,
        service_name=service_name,
        path=path,
        method=method,
        status_code=status_code,
        ip_address=ip_address,
        response_time_ms=response_time_ms,
        requested_at=requested_at,
        user_agent=user_agent,
        # Hub only needs request metadata. Never persist headers or bodies here:
        # AI responses can contain biometric embeddings and other private data.
        request_headers="",
        request_body="",
        response_headers="",
        response_body="",
        query_params=redact_query_params(query_params),
    )


def _do_send(
    *,
    url: str,
    secret: str,
    service_name: str,
    path: str,
    method: str,
    status_code: int,
    ip_address: Optional[str],
    response_time_ms: float,
    requested_at: datetime,
    user_agent: Optional[str],
    request_headers: str,
    request_body: str,
    response_headers: str,
    response_body: str,
    query_params: str,
) -> None:
    """실제 HTTP 전송 (ThreadPool 워커에서 실행)."""
    # path 는 최대 100000자, /로 시작해야 한다
    safe_path = ("/" + path.lstrip("/"))[:100_000] if path else "/"

    payload: dict = {
        "sourceService": service_name[:100],
        "path": safe_path,
        "method": method.upper()[:16],
        "statusCode": int(status_code),
        "success": 200 <= int(status_code) < 400,
        "ipAddress": _sanitize_ip(ip_address),
        # response_time_ms may be float (millisecond precision). Preserve sub-millisecond
        # precision by sending a float rounded to 3 decimal places (microsecond ~ 0.001 ms).
        "responseTimeMs": max(0.0, round(float(response_time_ms), 3)),
        # Note: only responseTimeMs (float) is sent. Microsecond field was removed.
        "requestedAt": _fmt_dt(requested_at),
        # Keep this invariant in the worker as well so direct calls cannot
        # accidentally forward biometric or credential material.
        "requestHeaders": "",
        "requestBody": "",
        "responseHeaders": "",
        "responseBody": "",
    }

    if user_agent is not None:
        payload["userAgent"] = user_agent

    if query_params:
        payload["queryParams"] = query_params[:2000]

    try:
        logger.debug(
            "[access-log] sending service=%s method=%s path=%s status=%s",
            payload["sourceService"],
            payload["method"],
            payload["path"],
            payload["statusCode"],
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib_request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "starsnap-access-log-sender/1.0")
        req.add_header("X-Hub-Log-Secret", secret)
        with urllib_request.urlopen(req, timeout=3.0) as resp:
            resp.read()  # 응답 소비 (커넥션 유지를 막기 위해)
    except urllib_error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        logger.warning(
            "[access-log] Server returned %s for %s body=%s",
            e.code,
            url,
            _truncate(redact_body(err_body), 1000),
        )
    except urllib_error.URLError as e:
        logger.warning("[access-log] Could not connect to log server %s: %s", url, e.reason)
    except Exception as e:  # noqa: BLE001
        logger.warning("[access-log] Unexpected error sending log: %s", e)
