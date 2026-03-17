"""Utility functions for langchain-google-classroom."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from langchain_core.utils import guard_import

if TYPE_CHECKING:
    from google.auth.credentials import Credentials as GoogleCredentials
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials as UserCredentials
    from google.oauth2.service_account import Credentials as ServiceCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google Classroom API scopes
# ---------------------------------------------------------------------------

CLASSROOM_COURSES_READONLY_SCOPE = (
    "https://www.googleapis.com/auth/classroom.courses.readonly"
)
CLASSROOM_COURSEWORK_READONLY_SCOPE = (
    "https://www.googleapis.com/auth/classroom.coursework.me.readonly"
)
CLASSROOM_ANNOUNCEMENTS_READONLY_SCOPE = (
    "https://www.googleapis.com/auth/classroom.announcements.readonly"
)
CLASSROOM_COURSEWORKMATERIALS_READONLY_SCOPE = (
    "https://www.googleapis.com/auth/classroom.courseworkmaterials.readonly"
)
DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"

DEFAULT_SCOPES: List[str] = [
    CLASSROOM_COURSES_READONLY_SCOPE,
    CLASSROOM_COURSEWORK_READONLY_SCOPE,
    CLASSROOM_ANNOUNCEMENTS_READONLY_SCOPE,
    CLASSROOM_COURSEWORKMATERIALS_READONLY_SCOPE,
    DRIVE_READONLY_SCOPE,
]

# ---------------------------------------------------------------------------
# Default credential file paths
# ---------------------------------------------------------------------------

DEFAULT_TOKEN_FILE = "token.json"
DEFAULT_CLIENT_SECRETS_FILE = "credentials.json"
DEFAULT_SERVICE_ACCOUNT_FILE = "service_account.json"


# ---------------------------------------------------------------------------
# Google library import helpers
# ---------------------------------------------------------------------------


def _import_google() -> Tuple[
    type[Request],
    type[UserCredentials],
    type[ServiceCredentials],
]:
    """Import google auth libraries.

    Returns:
        Tuple of ``Request``, user ``Credentials``, and ``ServiceCredentials``
        classes.
    """
    return (
        guard_import(
            module_name="google.auth.transport.requests",
            pip_name="google-auth",
        ).Request,
        guard_import(
            module_name="google.oauth2.credentials",
            pip_name="google-auth",
        ).Credentials,
        guard_import(
            module_name="google.oauth2.service_account",
            pip_name="google-auth",
        ).Credentials,
    )


def _import_installed_app_flow() -> type[InstalledAppFlow]:
    """Import ``InstalledAppFlow`` class.

    Returns:
        ``InstalledAppFlow`` class.
    """
    return guard_import(
        module_name="google_auth_oauthlib.flow",
        pip_name="google-auth-oauthlib",
    ).InstalledAppFlow


def _import_googleapiclient_build() -> Any:
    """Import ``googleapiclient.discovery.build`` function.

    Returns:
        ``googleapiclient.discovery.build`` function.
    """
    return guard_import(
        module_name="googleapiclient.discovery",
        pip_name="google-api-python-client",
    ).build


def _import_media_io_base_download() -> Any:
    """Import ``MediaIoBaseDownload`` for streaming file downloads.

    Returns:
        ``googleapiclient.http.MediaIoBaseDownload`` class.
    """
    return guard_import(
        module_name="googleapiclient.http",
        pip_name="google-api-python-client",
    ).MediaIoBaseDownload


def _import_http_error() -> Any:
    """Import ``HttpError`` for Google API error handling.

    Returns:
        ``googleapiclient.errors.HttpError`` class.
    """
    return guard_import(
        module_name="googleapiclient.errors",
        pip_name="google-api-python-client",
    ).HttpError


# ---------------------------------------------------------------------------
# Retry / backoff helper for Google API calls
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS_CODES = {429, 500, 503}
_MAX_RETRIES = 3
_BASE_DELAY = 1.0


def execute_with_retry(request: Any, *, max_retries: int = _MAX_RETRIES) -> Any:
    """Execute a Google API request with exponential backoff.

    Retries on HTTP 429 (rate limit), 500 (internal server error),
    and 503 (service unavailable).  Delay increases exponentially
    with jitter between attempts.

    Args:
        request: A ``googleapiclient`` HttpRequest object.
        max_retries: Maximum number of retry attempts.

    Returns:
        The API response dict.

    Raises:
        HttpError: If all retries are exhausted or a non-retryable error occurs.
    """
    import random
    import time

    HttpError = _import_http_error()

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return request.execute()
        except HttpError as exc:
            if exc.resp.status not in _RETRYABLE_STATUS_CODES:
                raise
            last_error = exc
            if attempt < max_retries:
                delay = _BASE_DELAY * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "API request failed (HTTP %s), retrying in %.1fs "
                    "(attempt %d/%d): %s",
                    exc.resp.status,
                    delay,
                    attempt + 1,
                    max_retries,
                    exc,
                )
                time.sleep(delay)
    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Credential helper — mirrors langchain_google_community._utils
# ---------------------------------------------------------------------------


def get_classroom_credentials(
    scopes: Optional[List[str]] = None,
    token_file: Optional[str] = None,
    client_secrets_file: Optional[str] = None,
    service_account_file: Optional[str] = None,
) -> GoogleCredentials:
    """Obtain Google credentials for Classroom API access.

    The resolution order is:

    1. Service account file (if provided and exists).
    2. Cached OAuth token (``token_file``).
    3. OAuth interactive login via ``InstalledAppFlow``.

    Args:
        scopes: API scopes to request.  Defaults to
            :data:`DEFAULT_SCOPES`.
        token_file: Path to cached OAuth token JSON.  Defaults to
            ``token.json``.
        client_secrets_file: Path to OAuth client-secrets JSON.  Defaults to
            ``credentials.json``.
        service_account_file: Path to a service-account key JSON.

    Returns:
        A ``google.oauth2.credentials.Credentials`` instance ready for API
        calls.
    """
    scopes = scopes or DEFAULT_SCOPES
    token_file = token_file or DEFAULT_TOKEN_FILE
    client_secrets_file = client_secrets_file or DEFAULT_CLIENT_SECRETS_FILE

    Request, UserCredentialsClass, ServiceCredentialsClass = _import_google()

    # 1. Service account -------------------------------------------------------
    if service_account_file and os.path.exists(service_account_file):
        logger.debug("Authenticating via service account: %s", service_account_file)
        return ServiceCredentialsClass.from_service_account_file(
            service_account_file, scopes=scopes
        )

    # 2. Cached token ----------------------------------------------------------
    creds: Optional[UserCredentials] = None
    if os.path.exists(token_file):
        creds = UserCredentialsClass.from_authorized_user_file(token_file, scopes)

    # 3. Refresh or interactive login ------------------------------------------
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            InstalledAppFlow = _import_installed_app_flow()
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, scopes
            )
            os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
            creds = flow.run_local_server(port=0)
        # Persist for next run
        with open(token_file, "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    return creds
