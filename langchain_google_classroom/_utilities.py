"""Utility functions for langchain-google-classroom."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from langchain_core.utils import guard_import

if TYPE_CHECKING:
    from google.auth.transport.requests import Request  
    from google.oauth2.credentials import Credentials  
    from google.oauth2.service_account import Credentials as ServiceCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow  

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google Classroom API scopes
# ---------------------------------------------------------------------------

CLASSROOM_COURSES_READONLY_SCOPE =  "https://www.googleapis.com/auth/classroom.courses.readonly"
CLASSROOM_COURSEWORK_READONLY_SCOPE = "https://www.googleapis.com/auth/classroom.coursework.me.readonly"
CLASSROOM_ANNOUNCEMENTS_READONLY_SCOPE = "https://www.googleapis.com/auth/classroom.announcements.readonly"
CLASSROOM_COURSEWORKMATERIALS_READONLY_SCOPE = "https://www.googleapis.com/auth/classroom.courseworkmaterials.readonly"


DEFAULT_SCOPES: List[str] = [
    CLASSROOM_COURSES_READONLY_SCOPE,
    CLASSROOM_COURSEWORK_READONLY_SCOPE,
    CLASSROOM_ANNOUNCEMENTS_READONLY_SCOPE,
    CLASSROOM_COURSEWORKMATERIALS_READONLY_SCOPE,
]

# ---------------------------------------------------------------------------
# Default credential file paths
# ---------------------------------------------------------------------------

DEFAULT_TOKEN_FILE = "token.json"
DEFAULT_CLIENT_SECRETS_FILE = "credentials.json"
DEFAULT_SERVICE_ACCOUNT_FILE = "service_account.json"


# ---------------------------------------------------------------------------
# Google library import helpers (guard_import pattern)
# ---------------------------------------------------------------------------


def _import_google() -> Tuple[Request, Credentials, ServiceCredentials]:
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


def _import_installed_app_flow() -> InstalledAppFlow:
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


# ---------------------------------------------------------------------------
# Credential helper — mirrors langchain_google_community._utils
# ---------------------------------------------------------------------------


def get_classroom_credentials(
    scopes: Optional[List[str]] = None,
    token_file: Optional[str] = None,
    client_secrets_file: Optional[str] = None,
    service_account_file: Optional[str] = None,
) -> Credentials:
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

    Request, Credentials, ServiceCredentials = _import_google()

    # 1. Service account -------------------------------------------------------
    if service_account_file and os.path.exists(service_account_file):
        logger.debug("Authenticating via service account: %s", service_account_file)
        return ServiceCredentials.from_service_account_file(
            service_account_file, scopes=scopes
        )

    # 2. Cached token ----------------------------------------------------------
    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, scopes)

    # 3. Refresh or interactive login ------------------------------------------
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # type: ignore[call-arg]
        else:
            InstalledAppFlow = _import_installed_app_flow()
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, scopes
            )
            creds = flow.run_local_server(port=0)
        # Persist for next run
        with open(token_file, "w") as token:
            token.write(creds.to_json())

    return creds