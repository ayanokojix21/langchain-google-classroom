"""Google Classroom document loader for LangChain."""

from __future__ import annotations

import asyncio
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
)

from langchain_core.document_loaders import BaseBlobParser, BaseLoader
from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from langchain_google_classroom._utilities import (
    DEFAULT_SCOPES,
    get_classroom_credentials,
)
from langchain_google_classroom.classroom_api import ClassroomAPIFetcher
from langchain_google_classroom.document_builder import (
    build_course_meta,
    build_from_announcement,
    build_from_attachment,
    build_from_course_work,
    build_from_link_attachment,
    build_from_material,
    build_from_rubric,
    build_from_student,
    build_from_submission,
    build_from_teacher,
    build_from_topic,
)
from langchain_google_classroom.normalizer import normalize
from langchain_google_classroom.parsers import get_parser

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional scopes — only requested when the corresponding feature is enabled
# ---------------------------------------------------------------------------

SUBMISSIONS_SCOPE = (
    "https://www.googleapis.com/auth/classroom.student-submissions.me.readonly"
)
ROSTERS_SCOPE = "https://www.googleapis.com/auth/classroom.rosters.readonly"
TOPICS_SCOPE = "https://www.googleapis.com/auth/classroom.topics.readonly"


class GoogleClassroomLoader(BaseLoader, BaseModel):
    """Load documents from Google Classroom.

    Inherits from
    [`BaseLoader`][langchain_core.document_loaders.BaseLoader] and
    [`BaseModel`][pydantic.BaseModel].

    Fetches courses, assignments (courseWork), announcements, course
    materials, student submissions, rubrics, topics, and roster data
    from the Google Classroom API and converts them into LangChain
    ``Document`` objects suitable for RAG pipelines, semantic search, and
    AI teaching assistants.

    When ``load_attachments`` is enabled, Drive file attachments on each
    classroom item are downloaded, parsed, and yielded as additional
    ``Document`` objects.

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

    ??? example "With Attachments"

        ```python
        loader = GoogleClassroomLoader(
            course_ids=["12345"],
            load_attachments=True,
            parse_attachments=True,
        )
        docs = loader.load()
        ```

    ??? example "Custom File Parser"

        ```python
        from langchain_community.document_loaders.parsers.pdf import (
            PyMuPDFParser,
        )

        loader = GoogleClassroomLoader(
            course_ids=["12345"],
            file_parser_cls=PyMuPDFParser,
        )
        docs = loader.load()
        ```

    ??? example "With Vision LLM (image understanding)"

        ```python
        from langchain_google_genai import ChatGoogleGenerativeAI

        loader = GoogleClassroomLoader(
            course_ids=["12345"],
            load_attachments=True,
            vision_model=ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
            ),
        )
        docs = loader.load()
        # PDF pages now include image understanding context
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

    ??? example "Load Student Submissions"

        ```python
        loader = GoogleClassroomLoader(
            course_ids=["12345"],
            load_submissions=True,
        )
        docs = loader.load()
        ```

    ??? example "Async Loading"

        ```python
        import asyncio
        from langchain_google_classroom import GoogleClassroomLoader


        async def main():
            loader = GoogleClassroomLoader(
                course_ids=["12345"],
            )
            docs = []
            async for doc in loader.alazy_load():
                docs.append(doc)
            return docs


        docs = asyncio.run(main())
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ------------------------------------------------------------------
    # Pydantic fields — replace the old __init__ parameters
    # ------------------------------------------------------------------

    # Content selection
    course_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "Specific course IDs to load. If None, all accessible courses are loaded."
        ),
    )
    load_assignments: bool = Field(
        default=True,
        description="Whether to load courseWork items.",
    )
    load_announcements: bool = Field(
        default=True,
        description="Whether to load announcements.",
    )
    load_materials: bool = Field(
        default=True,
        description="Whether to load courseWorkMaterials.",
    )
    load_attachments: bool = Field(
        default=True,
        description=(
            "Whether to resolve and load Drive file attachments on each classroom item."
        ),
    )
    parse_attachments: bool = Field(
        default=True,
        description=(
            "Whether to parse attachment file content using the parser "
            "layer. If False, raw bytes are decoded as UTF-8."
        ),
    )
    load_images: bool = Field(
        default=False,
        description=(
            "Whether to process image attachments. When False (default), "
            "image MIME types are skipped."
        ),
    )

    # New content types (Phase 3 — defaults to False for backward compat)
    load_submissions: bool = Field(
        default=False,
        description=(
            "Whether to load student submissions. Requires the "
            "classroom.student-submissions.me.readonly scope."
        ),
    )
    load_topics: bool = Field(
        default=False,
        description=(
            "Whether to load course topics. Requires the "
            "classroom.topics.readonly scope."
        ),
    )
    load_roster: bool = Field(
        default=False,
        description=(
            "Whether to load student and teacher roster. Requires the "
            "classroom.rosters.readonly scope."
        ),
    )

    # Vision / parser configuration
    vision_model: Optional[Any] = Field(
        default=None,
        description=(
            "Optional LangChain chat model with vision support. Passed to "
            "PDFParser and ImageParser for image understanding."
        ),
    )
    image_prompt: Optional[str] = Field(
        default=None,
        description=("Custom prompt sent to the vision model alongside each image."),
    )
    file_parser_cls: Optional[Any] = Field(
        default=None,
        description=(
            "Optional custom BaseBlobParser subclass to use for all "
            "attachment types. Bypasses the built-in MIME-type registry."
        ),
    )
    file_parser_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional keyword arguments forwarded to file_parser_cls when "
            "instantiating it."
        ),
    )

    # File size limit (Phase 4)
    max_file_size: int = Field(
        default=50_000_000,
        description=(
            "Maximum file size in bytes for attachments. Files larger than "
            "this are skipped. Default is 50 MB."
        ),
        ge=0,
    )

    # Authentication
    credentials: Optional[Any] = Field(
        default=None,
        exclude=True,
        description=(
            "Pre-built google.oauth2 credentials. When provided, the "
            "other credential arguments are ignored."
        ),
    )
    service_account_file: Optional[str] = Field(
        default=None,
        description="Path to a service-account key JSON file.",
    )
    token_file: Optional[str] = Field(
        default=None,
        description="Path to a cached OAuth token JSON file.",
    )
    client_secrets_file: Optional[str] = Field(
        default=None,
        description="Path to an OAuth client-secrets JSON file.",
    )
    scopes: List[str] = Field(
        default_factory=lambda: list(DEFAULT_SCOPES),
        description=(
            "API scopes to request. Defaults to read-only Classroom + "
            "Drive scopes. Additional scopes for submissions, roster, and "
            "topics are automatically added when those features are enabled."
        ),
    )

    # Internal — excluded from serialization / schema
    _parser_cache: Dict[str, BaseBlobParser] = PrivateAttr(default_factory=dict)

    # ClassVar to document which scopes are optional
    OPTIONAL_SCOPES: ClassVar[Dict[str, str]] = {
        "load_submissions": SUBMISSIONS_SCOPE,
        "load_roster": ROSTERS_SCOPE,
        "load_topics": TOPICS_SCOPE,
    }

    # ------------------------------------------------------------------
    # Pydantic validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _build_scopes(self) -> "GoogleClassroomLoader":
        """Automatically add opt-in scopes based on enabled features.

        Appends any additional scopes required by features that have been
        enabled (e.g. ``load_submissions``, ``load_roster``,
        ``load_topics``).
        """
        for flag, scope in self.OPTIONAL_SCOPES.items():
            if getattr(self, flag, False) and scope not in self.scopes:
                self.scopes.append(scope)

        return self

    # ------------------------------------------------------------------
    # BaseLoader interface
    # ------------------------------------------------------------------

    def lazy_load(self) -> Iterator[Document]:
        """Lazy-load documents from Google Classroom.

        Yields:
            ``Document`` objects for each assignment, announcement, or
            material found in the target courses, plus one or more
            ``Document`` objects per resolved attachment.
        """
        # 1. Obtain credentials ------------------------------------------------
        creds = self._get_credentials()

        # 2. Build API fetcher -------------------------------------------------
        fetcher = ClassroomAPIFetcher(credentials=creds)

        # 3. Build Drive resolver (only if attachments are enabled) ------------
        resolver = None
        if self.load_attachments:
            from langchain_google_classroom.drive_resolver import (
                DriveAttachmentResolver,
            )

            resolver = DriveAttachmentResolver(credentials=creds)

        # 4. Iterate courses ---------------------------------------------------
        for course in fetcher.list_courses(course_ids=self.course_ids):
            course_meta = build_course_meta(course)
            course_id = course.get("id", "")
            logger.info(
                "Processing course: %s (%s)",
                course_meta.get("course_name"),
                course_id,
            )

            # 4a. Assignments --------------------------------------------------
            if self.load_assignments:
                for item in fetcher.list_course_work(course_id):
                    yield build_from_course_work(item, course_meta)
                    if resolver:
                        yield from self._process_attachments(
                            resolver, item, course_meta, "assignment"
                        )
                        yield from self._process_link_attachments(
                            item, course_meta, "assignment"
                        )
                    # Fetch rubrics for this courseWork item
                    cw_id = item.get("id", "")
                    cw_title = item.get("title", "")
                    for rubric in fetcher.list_rubrics(course_id, cw_id):
                        yield build_from_rubric(
                            rubric, course_meta, course_work_title=cw_title
                        )

            # 4b. Announcements ------------------------------------------------
            if self.load_announcements:
                for item in fetcher.list_announcements(course_id):
                    yield build_from_announcement(item, course_meta)
                    if resolver:
                        yield from self._process_attachments(
                            resolver, item, course_meta, "announcement"
                        )
                        yield from self._process_link_attachments(
                            item, course_meta, "announcement"
                        )

            # 4c. Materials ----------------------------------------------------
            if self.load_materials:
                for item in fetcher.list_course_work_materials(course_id):
                    yield build_from_material(item, course_meta)
                    if resolver:
                        yield from self._process_attachments(
                            resolver, item, course_meta, "material"
                        )
                        yield from self._process_link_attachments(
                            item, course_meta, "material"
                        )

            # 4d. Student Submissions ------------------------------------------
            if self.load_submissions:
                for item in fetcher.list_student_submissions(course_id):
                    yield build_from_submission(item, course_meta)

            # 4e. Topics -------------------------------------------------------
            if self.load_topics:
                for item in fetcher.list_topics(course_id):
                    yield build_from_topic(item, course_meta)

            # 4f. Roster -------------------------------------------------------
            if self.load_roster:
                for item in fetcher.list_students(course_id):
                    yield build_from_student(item, course_meta)
                for item in fetcher.list_teachers(course_id):
                    yield build_from_teacher(item, course_meta)

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Async lazy-load documents from Google Classroom.

        Since ``google-api-python-client`` is synchronous, the entire
        ``lazy_load()`` generator is offloaded to a thread pool via
        ``asyncio.to_thread`` to avoid blocking the event loop.

        Yields:
            ``Document`` objects, same as :meth:`lazy_load`.
        """
        docs = await asyncio.to_thread(list, self.lazy_load())
        for doc in docs:
            yield doc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_parser_for(self, mime_type: str) -> Optional[BaseBlobParser]:
        """Return a parser for *mime_type*.

        If a custom ``file_parser_cls`` was provided at init time it is
        always used, regardless of MIME type.  Otherwise the built-in
        MIME-type registry is consulted.

        Image MIME types are skipped unless ``load_images`` is enabled.
        When a ``vision_model`` is configured it is injected into
        parsers that support it (``PDFParser``, ``ImageParser``).

        Args:
            mime_type: MIME type of the attachment content.

        Returns:
            A parser instance or ``None``.
        """
        if self.file_parser_cls is not None:
            return self.file_parser_cls(**self.file_parser_kwargs)

        normalized_mime = mime_type.split(";")[0].strip().lower()

        # Skip images unless explicitly enabled
        if normalized_mime.startswith("image/") and not self.load_images:
            return None

        parser: Optional[BaseBlobParser]
        cached = self._parser_cache.get(normalized_mime)
        if cached is not None:
            parser = cached
        else:
            parser = get_parser(normalized_mime)
            if parser is not None:
                self._parser_cache[normalized_mime] = parser

        # Inject vision model into parsers that support it
        if parser and self.vision_model:
            if hasattr(parser, "vision_model"):
                setattr(parser, "vision_model", self.vision_model)
                if self.image_prompt:
                    setattr(parser, "image_prompt", self.image_prompt)

        return parser

    def _process_link_attachments(
        self,
        item: Dict[str, Any],
        course_meta: Dict[str, Any],
        content_type: str,
    ) -> Iterator[Document]:
        """Yield Documents for YouTube and external link attachments.

        Unlike Drive files, these are not downloaded. Their metadata
        is captured into a structured Document.

        Args:
            item: Raw Classroom API item dict.
            course_meta: Dict with ``course_id`` and ``course_name``.
            content_type: Parent content type string.

        Yields:
            ``Document`` objects for each YouTube or link attachment.
        """
        from langchain_google_classroom.drive_resolver import (
            DriveAttachmentResolver,
        )

        for link_att in DriveAttachmentResolver.extract_link_attachments(item):
            yield build_from_link_attachment(
                url=link_att.url,
                title=link_att.title,
                attachment_type=link_att.attachment_type,
                extra=link_att.extra,
                parent_item=item,
                course_meta=course_meta,
                content_type=content_type,
            )

    def _process_attachments(
        self,
        resolver: Any,
        item: Dict[str, Any],
        course_meta: Dict[str, Any],
        content_type: str,
    ) -> Iterator[Document]:
        """Resolve, parse, and yield Documents for item attachments.

        Uses LangChain's :class:`~langchain_core.documents.base.Blob` to
        pass file content to parsers conforming to the
        :class:`~langchain_core.document_loaders.BaseBlobParser` interface.

        Args:
            resolver: A :class:`DriveAttachmentResolver` instance.
            item: Raw Classroom API item dict.
            course_meta: Dict with ``course_id`` and ``course_name``.
            content_type: Parent content type string.

        Yields:
            ``Document`` objects for each successfully parsed attachment.
        """
        parent_title = item.get("title", item.get("text", "")[:80])

        # Classroom-specific metadata to merge into parser output
        attachment_meta_base: Dict[str, Any] = {
            "source": "google_classroom",
            **course_meta,
            "item_id": item.get("id", ""),
            "parent_title": parent_title,
            "created_time": item.get("creationTime", ""),
            "updated_time": item.get("updateTime", ""),
            "alternate_link": item.get("alternateLink", ""),
        }

        for attachment in resolver.resolve(item, max_file_size=self.max_file_size):
            if self.parse_attachments:
                parser = self._get_parser_for(attachment.mime_type)
                if parser:
                    blob = Blob(
                        data=attachment.content,
                        mimetype=attachment.mime_type,
                        path=attachment.source_url,
                    )
                    try:
                        for doc in parser.lazy_parse(blob):
                            # Merge classroom metadata into parser output
                            doc.metadata.update(attachment_meta_base)
                            doc.metadata["content_type"] = f"{content_type}_attachment"
                            doc.metadata["title"] = attachment.title
                            doc.metadata["file_id"] = attachment.file_id
                            doc.metadata["mime_type"] = attachment.original_mime_type
                            doc.metadata["attachment_url"] = attachment.source_url
                            doc.page_content = normalize(doc.page_content)
                            yield doc
                    except Exception as exc:
                        logger.warning(
                            "Failed to parse attachment %s (%s): %s",
                            attachment.title,
                            attachment.mime_type,
                            exc,
                        )
                else:
                    logger.debug(
                        "No parser for MIME type %s, skipping %s",
                        attachment.mime_type,
                        attachment.title,
                    )
            else:
                # Raw decode when parse_attachments is disabled
                yield build_from_attachment(
                    file_id=attachment.file_id,
                    title=attachment.title,
                    mime_type=attachment.mime_type,
                    source_url=attachment.source_url,
                    original_mime_type=attachment.original_mime_type,
                    parsed_text=attachment.content.decode("utf-8", errors="replace"),
                    parent_item=item,
                    course_meta=course_meta,
                    content_type=content_type,
                )

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
