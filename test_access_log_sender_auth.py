from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

MODULE_NAME = "access_log_sender_under_test"
MODULE_PATH = Path(__file__).parent / "app" / "utils" / "access_log_sender.py"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC and SPEC.loader
access_log_sender = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = access_log_sender
SPEC.loader.exec_module(access_log_sender)

_do_send = access_log_sender._do_send
send_access_log = access_log_sender.send_access_log


def _payload():
    return {
        "url": "http://hub.test/api/server-logs",
        "secret": "shared-secret",
        "service_name": "starsnap-ai-backend",
        "path": "/api/health",
        "method": "GET",
        "status_code": 200,
        "ip_address": "127.0.0.1",
        "response_time_ms": 1.25,
        "requested_at": datetime.now(timezone.utc),
        "user_agent": "unit-test",
        "request_headers": "",
        "request_body": "",
        "response_headers": "",
        "response_body": "",
        "query_params": "",
    }


class AccessLogSenderAuthTest(unittest.TestCase):
    def test_sender_adds_hub_secret_header(self):
        response = MagicMock()
        response.__enter__.return_value = response

        with patch.object(access_log_sender.urllib_request, "urlopen", return_value=response) as urlopen:
            _do_send(**_payload())

        request = urlopen.call_args.args[0]
        self.assertEqual("shared-secret", request.get_header("X-hub-log-secret"))

    def test_blank_secret_disables_background_submission(self):
        payload = _payload()
        payload["secret"] = ""

        with patch.object(access_log_sender._executor, "submit") as submit:
            send_access_log(**payload)

        submit.assert_not_called()

    def test_worker_never_forwards_headers_or_bodies(self):
        payload = _payload()
        payload.update({
            "request_headers": "Authorization: Bearer private",
            "request_body": '{"password":"private"}',
            "response_headers": "Set-Cookie: private",
            "response_body": '{"embedding":[0.1,0.2],"token":"private"}',
        })
        response = MagicMock()
        response.__enter__.return_value = response

        with patch.object(access_log_sender.urllib_request, "urlopen", return_value=response) as urlopen:
            _do_send(**payload)

        request = urlopen.call_args.args[0]
        body = access_log_sender.json.loads(request.data.decode("utf-8"))
        self.assertEqual("", body["requestHeaders"])
        self.assertEqual("", body["requestBody"])
        self.assertEqual("", body["responseHeaders"])
        self.assertEqual("", body["responseBody"])


if __name__ == "__main__":
    unittest.main()
