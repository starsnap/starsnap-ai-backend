"""HTTP forwarding helpers.

이 모듈은 범용 HTTP 포워딩 함수(`forward_request`)와
multipart/form-data 바디를 생성하는 헬퍼(`build_multipart_payload`)를 제공합니다.

주요 함수들:
- forward_request: 모든 HTTP 메서드(GET/POST/PUT/DELETE 등)를 전달하고 응답(바디, 상태코드, 헤더)을 반환합니다.
- build_multipart_payload: multipart/form-data 바디와 바운더리를 생성합니다.
- forward_multipart_request: 기존 호환성용 래퍼로, multipart 바디를 전송합니다.
"""
from urllib import request as urllib_request, error as urllib_error
import json
from typing import Optional, Tuple, Dict, Any
import uuid
import mimetypes


def build_multipart_payload(
    *,
    file_bytes: bytes,
    filename: str,
    file_field_name: str = "file",
    fields: Optional[Dict[str, str]] = None,
    content_type: Optional[str] = None,
) -> Tuple[bytes, str]:
    """Create multipart/form-data body and boundary.

    Returns (body_bytes, boundary_string).
    """
    boundary = f"----starsnap-{uuid.uuid4().hex}"
    body = bytearray()
    text_fields = fields or {}

    for key, value in text_fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode("utf-8")
        )

    detected_content_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field_name}"; '
            f'filename="{filename}"\r\n'
        ).encode("utf-8")
    )
    body.extend(f"Content-Type: {detected_content_type}\r\n\r\n".encode("utf-8"))
    body.extend(file_bytes)
    body.extend("\r\n".encode("utf-8"))
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    return bytes(body), boundary


def forward_request(
    url: str,
    method: str = "GET",
    body: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = 5.0,
    expect_json: bool = False,
    error_prefix: str = "upstream",
) -> Tuple[Optional[Any], Optional[int], Dict[str, str]]:
    """Forward a generic HTTP request to `url`.

    Returns (response_body_or_parsed_json_or_error_dict, status_code, response_headers).
    On connection error returns (error_dict, 502, {}).
    """
    hdrs = dict(headers or {})
    req = urllib_request.Request(url, data=body, headers=hdrs, method=method.upper())

    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            resp_body = resp.read()
            resp_headers = {k: v for k, v in resp.getheaders()}
            if expect_json:
                try:
                    return json.loads(resp_body.decode("utf-8")), resp.getcode(), resp_headers
                except Exception:
                    # fall back to raw bytes if JSON parsing fails
                    return resp_body, resp.getcode(), resp_headers
            return resp_body, resp.getcode(), resp_headers
    except urllib_error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        try:
            parsed_error = json.loads(error_body) if error_body else {"error": "upstream http error"}
        except json.JSONDecodeError:
            parsed_error = {"error": "upstream http error", "upstream_body": error_body}
        return parsed_error, e.code, {}
    except urllib_error.URLError:
        return {"error": f"failed to connect upstream {error_prefix} api"}, 502, {}


def forward_multipart_request(
    url: str,
    multipart_body: bytes,
    boundary: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = 5.0,
    error_prefix: str = "photo",
) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    """Compatibility wrapper for sending multipart/form-data requests.

    Returns (None, None) on success, otherwise (parsed_error_dict, status_code).
    """
    hdrs = dict(headers or {})
    hdrs["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    body, status, resp_headers = forward_request(
        url=url,
        method="POST",
        body=multipart_body,
        headers=hdrs,
        timeout=timeout,
        expect_json=False,
        error_prefix=error_prefix,
    )
    # If body is dict and status >=400, it's an error parsed by forward_request
    if status is None:
        return {"error": "no response from upstream"}, None
    if status >= 400:
        return body if isinstance(body, dict) else {"error": "upstream http error", "upstream_body": body}, status
    return None, None


