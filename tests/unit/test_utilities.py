"""Unit tests for authentication and retry utilities."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_google_classroom._utilities import (
    DEFAULT_SCOPES,
    DRIVE_READONLY_SCOPE,
    execute_with_retry,
    get_classroom_credentials,
)

# ---------------------------------------------------------------------------
# Helper — mock HttpError class
# ---------------------------------------------------------------------------


def _make_http_error_class() -> type:
    """Create a mock HttpError class for testing."""

    class MockHttpError(Exception):
        def __init__(self, resp: Any, content: bytes) -> None:
            self.resp = resp
            self.content = content
            super().__init__(f"HTTP {resp.status}")

    return MockHttpError


# ---------------------------------------------------------------------------
# Tests — execute_with_retry
# ---------------------------------------------------------------------------


class TestExecuteWithRetry:
    """Tests for the exponential-backoff retry helper."""

    def test_success_first_attempt(self) -> None:
        """A successful request should return immediately."""
        mock_request = MagicMock()
        mock_request.execute.return_value = {"id": "123"}

        result = execute_with_retry(mock_request)

        assert result == {"id": "123"}
        assert mock_request.execute.call_count == 1

    @patch("time.sleep")
    def test_retry_on_429(self, mock_sleep: MagicMock) -> None:
        """HTTP 429 should be retried."""
        HttpError = _make_http_error_class()

        mock_request = MagicMock()
        mock_request.execute.side_effect = [
            HttpError(resp=MagicMock(status=429), content=b"rate limit"),
            {"id": "123"},
        ]

        with patch(
            "langchain_google_classroom._utilities._import_http_error",
            return_value=HttpError,
        ):
            result = execute_with_retry(mock_request, max_retries=3)

        assert result == {"id": "123"}
        assert mock_request.execute.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("time.sleep")
    def test_retry_on_503(self, mock_sleep: MagicMock) -> None:
        """HTTP 503 should be retried."""
        HttpError = _make_http_error_class()

        mock_request = MagicMock()
        mock_request.execute.side_effect = [
            HttpError(
                resp=MagicMock(status=503),
                content=b"unavailable",
            ),
            HttpError(
                resp=MagicMock(status=503),
                content=b"unavailable",
            ),
            {"ok": True},
        ]

        with patch(
            "langchain_google_classroom._utilities._import_http_error",
            return_value=HttpError,
        ):
            result = execute_with_retry(mock_request, max_retries=3)

        assert result == {"ok": True}
        assert mock_request.execute.call_count == 3
        assert mock_sleep.call_count == 2

    def test_non_retryable_error_raises(self) -> None:
        """HTTP 403 (non-retryable) should raise immediately."""
        HttpError = _make_http_error_class()

        mock_request = MagicMock()
        mock_request.execute.side_effect = HttpError(
            resp=MagicMock(status=403), content=b"forbidden"
        )

        with patch(
            "langchain_google_classroom._utilities._import_http_error",
            return_value=HttpError,
        ):
            with pytest.raises(HttpError):
                execute_with_retry(mock_request)

        assert mock_request.execute.call_count == 1

    @patch("time.sleep")
    def test_exhausted_retries_raises(self, mock_sleep: MagicMock) -> None:
        """When all retries are exhausted, the last error is raised."""
        HttpError = _make_http_error_class()
        error = HttpError(resp=MagicMock(status=500), content=b"error")

        mock_request = MagicMock()
        mock_request.execute.side_effect = error

        with patch(
            "langchain_google_classroom._utilities._import_http_error",
            return_value=HttpError,
        ):
            with pytest.raises(HttpError):
                execute_with_retry(mock_request, max_retries=2)

        # initial attempt + 2 retries = 3 calls
        assert mock_request.execute.call_count == 3


# ---------------------------------------------------------------------------
# Tests — get_classroom_credentials
# ---------------------------------------------------------------------------


class TestGetClassroomCredentials:
    """Tests for the credential resolution logic."""

    @patch("langchain_google_classroom._utilities._import_google")
    def test_service_account_path(
        self, mock_import_google: MagicMock, tmp_path: Any
    ) -> None:
        """Service account file should be used when it exists."""
        sa_file = tmp_path / "sa.json"
        sa_file.write_text("{}")

        mock_request_cls = MagicMock()
        mock_user_creds_cls = MagicMock()
        mock_sa_creds_cls = MagicMock()
        mock_creds = MagicMock()
        mock_sa_creds_cls.from_service_account_file.return_value = mock_creds

        mock_import_google.return_value = (
            mock_request_cls,
            mock_user_creds_cls,
            mock_sa_creds_cls,
        )

        result = get_classroom_credentials(service_account_file=str(sa_file))

        assert result is mock_creds
        mock_sa_creds_cls.from_service_account_file.assert_called_once()

    @patch("langchain_google_classroom._utilities._import_google")
    def test_cached_token_valid(
        self, mock_import_google: MagicMock, tmp_path: Any
    ) -> None:
        """A valid cached token should be used directly."""
        token_file = tmp_path / "token.json"
        token_file.write_text("{}")

        mock_request_cls = MagicMock()
        mock_user_creds_cls = MagicMock()
        mock_sa_creds_cls = MagicMock()

        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_user_creds_cls.from_authorized_user_file.return_value = mock_creds

        mock_import_google.return_value = (
            mock_request_cls,
            mock_user_creds_cls,
            mock_sa_creds_cls,
        )

        result = get_classroom_credentials(
            token_file=str(token_file),
            service_account_file="nonexistent_sa.json",
        )

        assert result is mock_creds

    @patch("langchain_google_classroom._utilities._import_installed_app_flow")
    @patch("langchain_google_classroom._utilities._import_google")
    def test_expired_token_refreshes(
        self,
        mock_import_google: MagicMock,
        mock_import_flow: MagicMock,
        tmp_path: Any,
    ) -> None:
        """An expired token with refresh_token should be refreshed."""
        token_file = tmp_path / "token.json"
        token_file.write_text("{}")

        mock_request_cls = MagicMock()
        mock_user_creds_cls = MagicMock()
        mock_sa_creds_cls = MagicMock()

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "refresh_abc"
        mock_creds.to_json.return_value = "{}"
        mock_user_creds_cls.from_authorized_user_file.return_value = mock_creds

        mock_import_google.return_value = (
            mock_request_cls,
            mock_user_creds_cls,
            mock_sa_creds_cls,
        )

        result = get_classroom_credentials(
            token_file=str(token_file),
            service_account_file="nonexistent_sa.json",
        )

        assert result is mock_creds
        mock_creds.refresh.assert_called_once()

    @patch("langchain_google_classroom._utilities._import_installed_app_flow")
    @patch("langchain_google_classroom._utilities._import_google")
    def test_oauth_flow_fallback(
        self,
        mock_import_google: MagicMock,
        mock_import_flow: MagicMock,
        tmp_path: Any,
    ) -> None:
        """When no cached token exists, OAuth flow should run."""
        token_file = tmp_path / "token.json"
        client_secrets = tmp_path / "creds.json"
        client_secrets.write_text("{}")

        mock_request_cls = MagicMock()
        mock_user_creds_cls = MagicMock()
        mock_sa_creds_cls = MagicMock()

        mock_import_google.return_value = (
            mock_request_cls,
            mock_user_creds_cls,
            mock_sa_creds_cls,
        )

        mock_flow = MagicMock()
        mock_new_creds = MagicMock()
        mock_new_creds.to_json.return_value = "{}"
        mock_flow.run_local_server.return_value = mock_new_creds
        mock_import_flow.return_value = MagicMock()
        mock_import_flow.return_value.from_client_secrets_file.return_value = mock_flow

        result = get_classroom_credentials(
            token_file=str(token_file),
            client_secrets_file=str(client_secrets),
            service_account_file="nonexistent_sa.json",
        )

        assert result is mock_new_creds
        mock_flow.run_local_server.assert_called_once_with(port=0)
        assert token_file.exists()

    def test_default_scopes_include_drive(self) -> None:
        """DEFAULT_SCOPES should include the Drive readonly scope."""
        assert DRIVE_READONLY_SCOPE in DEFAULT_SCOPES
        assert len(DEFAULT_SCOPES) == 5
