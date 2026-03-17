"""Google Classroom document loader for LangChain."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Type

from langchain_core.document_loaders import BaseBlobParser, BaseLoader
from langchain_core.documents import Document
from langchain_core.documents.base import Blob

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
    build_from_material,
)
from langchain_google_classroom.normalizer import normalize
from langchain_google_classroom.parsers import get_parser

logger = logging.getLogger(__name__)


class GoogleClassroomLoader(BaseLoader):
    """Load documents from Google Classroom.

    Inherits from
    [`BaseLoader`][langchain_core.document_loaders.BaseLoader].

    Fetches courses, assignments (courseWork), announcements, and course
    materials from the Google Classroom API and converts them into LangChain
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
    """

    def __init__(
        self,
        course_ids: Optional[List[str]] = None,
        *,
        load_assignments: bool = True,
        load_announcements: bool = True,
        load_materials: bool = True,
        load_attachments: bool = True,
        parse_attachments: bool = True,
        load_images: bool = False,
        vision_model: Optional[Any] = None,
        image_prompt: Optional[str] = None,
        file_parser_cls: Optional[Type[BaseBlobParser]] = None,
        file_parser_kwargs: Optional[Dict[str, Any]] = None,
        credentials: Optional[Any] = None,
        service_account_file: Optional[str] = None,
        token_file: Optional[str] = None,
        client_secrets_file: Optional[str] = None,
        scopes: Optional[List[str]] = None,
    ) -> None:
        """Initialise the loader.

        Args:
            course_ids: Specific course IDs to load.  If ``None``, all
                accessible courses are loaded.
            load_assignments: Whether to load courseWork items.
            load_announcements: Whether to load announcements.
            load_materials: Whether to load courseWorkMaterials.
            load_attachments: Whether to resolve and load Drive file
                attachments on each classroom item.
            parse_attachments: Whether to parse attachment file content
                using the parser layer.  If ``False``, raw bytes are
                decoded as UTF-8.
            load_images: Whether to process image attachments.  When
                ``False`` (default), image MIME types are skipped.
            vision_model: Optional LangChain chat model with vision
                support (e.g. ``ChatGoogleGenerativeAI``,
                ``ChatOpenAI`` with GPT-4V).  Passed to
                ``PDFParser`` and ``ImageParser`` so embedded images
                are described by the model.
            image_prompt: Custom prompt sent to the vision model
                alongside each image.  Defaults to a student-friendly
                description prompt.
            file_parser_cls: Optional custom
                :class:`~langchain_core.document_loaders.BaseBlobParser`
                subclass to use for **all** attachment types.  When set,
                the built-in MIME-type registry is bypassed and this
                parser handles every file.  This lets users plug in
                ``PyMuPDFParser``, ``UnstructuredParser``, etc.
            file_parser_kwargs: Optional keyword arguments forwarded to
                *file_parser_cls* when instantiating it.
            credentials: Pre-built ``google.oauth2`` credentials.  When
                provided the other credential arguments are ignored.
            service_account_file: Path to a service-account key JSON file.
            token_file: Path to a cached OAuth token JSON file.
            client_secrets_file: Path to an OAuth client-secrets JSON file.
            scopes: API scopes to request.  Defaults to read-only
                Classroom + Drive scopes.
        """
        self.course_ids = course_ids
        self.load_assignments = load_assignments
        self.load_announcements = load_announcements
        self.load_materials = load_materials
        self.load_attachments = load_attachments
        self.parse_attachments = parse_attachments
        self.load_images = load_images
        self.vision_model = vision_model
        self.image_prompt = image_prompt
        self.file_parser_cls = file_parser_cls
        self.file_parser_kwargs = file_parser_kwargs or {}
        self._parser_cache: Dict[str, BaseBlobParser] = {}
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

            # 4b. Announcements ------------------------------------------------
            if self.load_announcements:
                for item in fetcher.list_announcements(course_id):
                    yield build_from_announcement(item, course_meta)
                    if resolver:
                        yield from self._process_attachments(
                            resolver, item, course_meta, "announcement"
                        )

            # 4c. Materials ----------------------------------------------------
            if self.load_materials:
                for item in fetcher.list_course_work_materials(course_id):
                    yield build_from_material(item, course_meta)
                    if resolver:
                        yield from self._process_attachments(
                            resolver, item, course_meta, "material"
                        )

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

        for attachment in resolver.resolve(item):
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
