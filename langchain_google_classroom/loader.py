"""Google Classroom document loader for LangChain."""

from __future__ import annotations

import logging
from typing import Any, Iterator, List, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_google_classroom._utilities import (
    DEFAULT_SCOPES,
    get_classroom_credentials,
)
from langchain_google_classroom.classroom_api import ClassroomAPIFetcher
from langchain_google_classroom.document_builder import (
    build_course_meta,
    build_from_announcement,
    build_from_course_work,
    build_from_material,
)

logger = logging.getLogger(__name__)


class GoogleClassroomLoader(BaseLoader):
    """Load documents from Google Classroom.

    Inherits from
    [`BaseLoader`][langchain_core.document_loaders.BaseLoader].

    Fetches courses, assignments (courseWork), announcements, and course
    materials from the Google Classroom API and converts them into LangChain
    ``Document`` objects suitable for RAG pipelines, semantic search, and
    AI teaching assistants.

    !!! note "Installation"

        ```bash
        pip install langchain-google-classroom
        ```

    !!! note "Authentication"

        Requires Google Cloud credentials with Classroom API enabled.
        Supports service-account keys, OAuth user credentials, and
        pre-built credential objects.

    ??? example "Basic Usage"

        ```python
        from langchain_google_classroom import GoogleClassroomLoader

        loader = GoogleClassroomLoader(course_ids=["12345"])
        docs = loader.load()
        ```

    ??? example "Service Account"

        ```python
        loader = GoogleClassroomLoader(
            service_account_file="service_account.json",
        )
        docs = loader.load()
        ```

    ??? example "Selective Loading"

        ```python
        loader = GoogleClassroomLoader(
            course_ids=["12345"],
            load_assignments=True,
            load_announcements=False,
            load_materials=False,
        )
        docs = loader.load()
        ```
    """

    def __init__(
        self,
        course_ids: Optional[List[str]] = None,
        *,
        load_assignments: bool = True,
        load_announcements: bool = True,
        load_materials: bool = True,
        credentials: Optional[Any] = None,
        service_account_file: Optional[str] = None,
        token_file: Optional[str] = None,
        client_secrets_file: Optional[str] = None,
        scopes: Optional[List[str]] = None,
    ) -> None:
        """Initialise the loader.

        Args:
            course_ids: Specific course IDs to load.  If ``None``, all accessible courses are loaded.
            load_assignments: Whether to load courseWork items.
            load_announcements: Whether to load announcements.
            load_materials: Whether to load courseWorkMaterials.
            credentials: Pre-built ``google.oauth2`` credentials.  When provided the other credential arguments are ignored.
            service_account_file: Path to a service-account key JSON file.
            token_file: Path to a cached OAuth token JSON file.
            client_secrets_file: Path to an OAuth client-secrets JSON file.
            scopes: API scopes to request.  Defaults to read-only Classroom scopes.
        """
        self.course_ids = course_ids
        self.load_assignments = load_assignments
        self.load_announcements = load_announcements
        self.load_materials = load_materials
        self.credentials = credentials
        self.service_account_file = service_account_file
        self.token_file = token_file
        self.client_secrets_file = client_secrets_file
        self.scopes = scopes or DEFAULT_SCOPES

    # ------------------------------------------------------------------
    # BaseLoader interface
    # ------------------------------------------------------------------

    def lazy_load(self) -> Iterator[Document]:
        """Lazy-load documents from Google Classroom.

        Yields:
            ``Document`` objects for each assignment, announcement, or
            material found in the target courses.
        """
        # 1. Obtain credentials ------------------------------------------------
        creds = self._get_credentials()

        # 2. Build API fetcher -------------------------------------------------
        fetcher = ClassroomAPIFetcher(credentials=creds)

        # 3. Iterate courses ---------------------------------------------------
        for course in fetcher.list_courses(course_ids=self.course_ids):
            course_meta = build_course_meta(course)
            course_id = course.get("id", "")
            logger.info(
                "Processing course: %s (%s)",
                course_meta.get("course_name"),
                course_id,
            )

            # 3a. Assignments --------------------------------------------------
            if self.load_assignments:
                for item in fetcher.list_course_work(course_id):
                    yield build_from_course_work(item, course_meta)

            # 3b. Announcements ------------------------------------------------
            if self.load_announcements:
                for item in fetcher.list_announcements(course_id):
                    yield build_from_announcement(item, course_meta)

            # 3c. Materials ----------------------------------------------------
            if self.load_materials:
                for item in fetcher.list_course_work_materials(course_id):
                    yield build_from_material(item, course_meta)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_credentials(self) -> Any:
        """Resolve credentials using the configured strategy.

        Returns:
            A credentials object ready for Google API calls.
        """
        # If pre-built credentials were provided, use them directly.
        if self.credentials is not None:
            return self.credentials

        return get_classroom_credentials(
            scopes=self.scopes,
            token_file=self.token_file,
            client_secrets_file=self.client_secrets_file,
            service_account_file=self.service_account_file,
        )
