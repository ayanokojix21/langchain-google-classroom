"""Unit tests for GoogleClassroomLoader."""

from __future__ import annotations

from typing import Any, Dict, Iterator
from unittest.mock import MagicMock, patch

from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents import Document
from langchain_core.documents.base import Blob

from langchain_google_classroom.loader import GoogleClassroomLoader

# ---------------------------------------------------------------------------
# Fixtures — sample API responses
# ---------------------------------------------------------------------------

SAMPLE_COURSE: Dict[str, Any] = {
    "id": "12345",
    "name": "Machine Learning",
    "courseState": "ACTIVE",
}

SAMPLE_COURSE_WORK_1: Dict[str, Any] = {
    "id": "cw_001",
    "title": "Homework 1",
    "description": "Complete exercises 1-5 from Chapter 2.",
    "state": "PUBLISHED",
    "creationTime": "2024-01-10T08:00:00Z",
    "updateTime": "2024-01-10T08:00:00Z",
    "alternateLink": "https://classroom.google.com/c/12345/a/cw_001",
    "maxPoints": 100.0,
    "dueDate": {"year": 2024, "month": 1, "day": 20},
    "dueTime": {"hours": 23, "minutes": 59},
}

SAMPLE_COURSE_WORK_2: Dict[str, Any] = {
    "id": "cw_002",
    "title": "Homework 2",
    "description": "Read Chapter 3.",
    "state": "PUBLISHED",
    "creationTime": "2024-01-15T08:00:00Z",
    "updateTime": "2024-01-15T08:00:00Z",
    "alternateLink": "https://classroom.google.com/c/12345/a/cw_002",
}

SAMPLE_COURSE_WORK_WITH_ATTACHMENT: Dict[str, Any] = {
    "id": "cw_003",
    "title": "Homework 3",
    "description": "See attached PDF.",
    "state": "PUBLISHED",
    "creationTime": "2024-01-20T08:00:00Z",
    "updateTime": "2024-01-20T08:00:00Z",
    "alternateLink": "https://classroom.google.com/c/12345/a/cw_003",
    "materials": [
        {
            "driveFile": {
                "driveFile": {
                    "id": "file_aaa",
                    "title": "Instructions.pdf",
                    "alternateLink": "https://drive.google.com/file/d/file_aaa/view",
                },
            }
        }
    ],
}

SAMPLE_ANNOUNCEMENT: Dict[str, Any] = {
    "id": "ann_001",
    "text": "Welcome to Machine Learning! Please review the syllabus.",
    "state": "PUBLISHED",
    "creationTime": "2024-01-05T10:00:00Z",
    "updateTime": "2024-01-05T10:00:00Z",
    "alternateLink": "https://classroom.google.com/c/12345/a/ann_001",
}

SAMPLE_MATERIAL: Dict[str, Any] = {
    "id": "mat_001",
    "title": "Lecture 1 Slides",
    "description": "Introduction to supervised learning.",
    "state": "PUBLISHED",
    "creationTime": "2024-01-08T09:00:00Z",
    "updateTime": "2024-01-08T09:00:00Z",
    "alternateLink": "https://classroom.google.com/c/12345/a/mat_001",
}


# ---------------------------------------------------------------------------
# Helper to build a loader with mocked credentials
# ---------------------------------------------------------------------------


def _make_loader(**kwargs: Any) -> GoogleClassroomLoader:
    """Create a loader with a dummy credentials object."""
    return GoogleClassroomLoader(
        credentials=MagicMock(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests — core functionality (attachments disabled)
# ---------------------------------------------------------------------------


class TestGoogleClassroomLoader:
    """Tests for the GoogleClassroomLoader class."""

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_lazy_load_assignments(self, mock_fetcher_cls: MagicMock) -> None:
        """Two courseWork items should produce two Documents."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter(
            [SAMPLE_COURSE_WORK_1, SAMPLE_COURSE_WORK_2]
        )
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])

        loader = _make_loader(course_ids=["12345"], load_attachments=False)
        docs = loader.load()

        assert len(docs) == 2
        assert docs[0].metadata["content_type"] == "assignment"
        assert docs[0].metadata["title"] == "Homework 1"
        assert docs[0].metadata["course_id"] == "12345"
        assert docs[0].metadata["course_name"] == "Machine Learning"
        assert docs[0].metadata["source"] == "google_classroom"
        assert "due_date" in docs[0].metadata
        assert docs[0].metadata["due_date"] == "2024-01-20T23:59:00"
        assert docs[0].metadata["max_points"] == 100.0
        assert "Assignment: Homework 1" in docs[0].page_content

        assert docs[1].metadata["title"] == "Homework 2"
        assert "due_date" not in docs[1].metadata

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_lazy_load_announcements(self, mock_fetcher_cls: MagicMock) -> None:
        """One announcement should produce one Document."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter([])
        fetcher.list_announcements.return_value = iter([SAMPLE_ANNOUNCEMENT])
        fetcher.list_course_work_materials.return_value = iter([])

        loader = _make_loader(course_ids=["12345"], load_attachments=False)
        docs = loader.load()

        assert len(docs) == 1
        doc = docs[0]
        assert doc.metadata["content_type"] == "announcement"
        assert "Welcome to Machine Learning" in doc.page_content
        assert doc.metadata["item_id"] == "ann_001"

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_lazy_load_materials(self, mock_fetcher_cls: MagicMock) -> None:
        """One material should produce one Document."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter([])
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([SAMPLE_MATERIAL])

        loader = _make_loader(course_ids=["12345"], load_attachments=False)
        docs = loader.load()

        assert len(docs) == 1
        doc = docs[0]
        assert doc.metadata["content_type"] == "material"
        assert "Lecture 1 Slides" in doc.page_content

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_selective_loading_no_announcements(
        self, mock_fetcher_cls: MagicMock
    ) -> None:
        """Setting ``load_announcements=False`` should skip announcements."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter([SAMPLE_COURSE_WORK_1])
        fetcher.list_course_work_materials.return_value = iter([])

        loader = _make_loader(
            course_ids=["12345"],
            load_announcements=False,
            load_attachments=False,
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["content_type"] == "assignment"
        fetcher.list_announcements.assert_not_called()

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_selective_loading_only_announcements(
        self, mock_fetcher_cls: MagicMock
    ) -> None:
        """Load only announcements."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_announcements.return_value = iter([SAMPLE_ANNOUNCEMENT])

        loader = _make_loader(
            course_ids=["12345"],
            load_assignments=False,
            load_announcements=True,
            load_materials=False,
            load_attachments=False,
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["content_type"] == "announcement"
        fetcher.list_course_work.assert_not_called()
        fetcher.list_course_work_materials.assert_not_called()

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_empty_course(self, mock_fetcher_cls: MagicMock) -> None:
        """A course with no items should produce no Documents."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter([])
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])

        loader = _make_loader(course_ids=["12345"], load_attachments=False)
        docs = loader.load()

        assert docs == []

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_no_courses(self, mock_fetcher_cls: MagicMock) -> None:
        """No accessible courses should produce no Documents."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([])

        loader = _make_loader(load_attachments=False)
        docs = loader.load()

        assert docs == []

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_multiple_courses(self, mock_fetcher_cls: MagicMock) -> None:
        """Documents from multiple courses should all be yielded."""
        course_a = {"id": "aaa", "name": "Course A"}
        course_b = {"id": "bbb", "name": "Course B"}

        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([course_a, course_b])

        def _course_work_side_effect(course_id: str):  # type: ignore[no-untyped-def]
            if course_id == "aaa":
                return iter([SAMPLE_COURSE_WORK_1])
            return iter([SAMPLE_COURSE_WORK_2])

        fetcher.list_course_work.side_effect = _course_work_side_effect
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])

        loader = _make_loader(course_ids=["aaa", "bbb"], load_attachments=False)
        docs = loader.load()

        assert len(docs) == 2
        assert docs[0].metadata["course_name"] == "Course A"
        assert docs[1].metadata["course_name"] == "Course B"

    def test_credentials_passthrough(self) -> None:
        """Pre-built credentials should be used directly."""
        creds = MagicMock()
        loader = GoogleClassroomLoader(credentials=creds)
        assert loader._get_credentials() is creds


# ---------------------------------------------------------------------------
# Tests — attachment integration
# ---------------------------------------------------------------------------


class TestGoogleClassroomLoaderAttachments:
    """Tests for attachment loading in GoogleClassroomLoader."""

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_attachments_disabled_no_resolver(
        self, mock_fetcher_cls: MagicMock
    ) -> None:
        """When load_attachments=False, no attachment Documents are yielded."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter(
            [SAMPLE_COURSE_WORK_WITH_ATTACHMENT]
        )
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])

        loader = _make_loader(course_ids=["12345"], load_attachments=False)
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["content_type"] == "assignment"

    @patch(
        "langchain_google_classroom.loader.get_parser",
    )
    @patch(
        "langchain_google_classroom.drive_resolver.DriveAttachmentResolver",
        autospec=True,
    )
    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_attachments_enabled_yields_attachment_doc(
        self,
        mock_fetcher_cls: MagicMock,
        mock_resolver_cls: MagicMock,
        mock_get_parser: MagicMock,
    ) -> None:
        """Assignment + 1 attachment should yield 2 Documents."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter(
            [SAMPLE_COURSE_WORK_WITH_ATTACHMENT]
        )
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])

        from langchain_google_classroom.drive_resolver import (
            ResolvedAttachment,
        )

        mock_attachment = ResolvedAttachment(
            file_id="file_aaa",
            title="Instructions.pdf",
            mime_type="text/plain",
            content=b"Parsed PDF text content here",
            source_url="https://drive.google.com/file/d/file_aaa/view",
            original_mime_type="application/pdf",
        )
        resolver = mock_resolver_cls.return_value
        resolver.resolve.return_value = iter([mock_attachment])

        # Mock parser: return a Document via lazy_parse
        mock_parser = MagicMock(spec=BaseBlobParser)
        mock_parser.lazy_parse.return_value = iter(
            [Document(page_content="Parsed PDF text content here", metadata={})]
        )
        mock_get_parser.return_value = mock_parser

        loader = _make_loader(
            course_ids=["12345"],
            load_attachments=True,
            parse_attachments=True,
        )
        docs = loader.load()

        assert len(docs) == 2
        # First doc = assignment
        assert docs[0].metadata["content_type"] == "assignment"
        assert docs[0].metadata["title"] == "Homework 3"
        # Second doc = attachment (metadata merged by loader)
        assert docs[1].metadata["content_type"] == "assignment_attachment"
        assert docs[1].metadata["title"] == "Instructions.pdf"
        assert docs[1].metadata["file_id"] == "file_aaa"
        assert docs[1].metadata["parent_title"] == "Homework 3"
        assert docs[1].metadata["source"] == "google_classroom"
        assert "Parsed PDF text content here" in docs[1].page_content

    @patch(
        "langchain_google_classroom.drive_resolver.DriveAttachmentResolver",
        autospec=True,
    )
    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_parse_disabled_raw_decode(
        self,
        mock_fetcher_cls: MagicMock,
        mock_resolver_cls: MagicMock,
    ) -> None:
        """With parse_attachments=False, content is raw-decoded via
        build_from_attachment."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter(
            [SAMPLE_COURSE_WORK_WITH_ATTACHMENT]
        )
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])

        from langchain_google_classroom.drive_resolver import (
            ResolvedAttachment,
        )

        mock_attachment = ResolvedAttachment(
            file_id="file_aaa",
            title="data.txt",
            mime_type="text/plain",
            content=b"Raw text content",
            source_url="https://drive.google.com/file/d/file_aaa/view",
            original_mime_type="text/plain",
        )
        resolver = mock_resolver_cls.return_value
        resolver.resolve.return_value = iter([mock_attachment])

        loader = _make_loader(
            course_ids=["12345"],
            load_attachments=True,
            parse_attachments=False,
        )
        docs = loader.load()

        assert len(docs) == 2
        assert docs[1].metadata["content_type"] == "assignment_attachment"
        assert "Raw text content" in docs[1].page_content

    @patch(
        "langchain_google_classroom.loader.get_parser",
    )
    @patch(
        "langchain_google_classroom.drive_resolver.DriveAttachmentResolver",
        autospec=True,
    )
    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_unsupported_mime_skipped(
        self,
        mock_fetcher_cls: MagicMock,
        mock_resolver_cls: MagicMock,
        mock_get_parser: MagicMock,
    ) -> None:
        """Attachments with unsupported MIME types should be skipped."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter(
            [SAMPLE_COURSE_WORK_WITH_ATTACHMENT]
        )
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])

        from langchain_google_classroom.drive_resolver import (
            ResolvedAttachment,
        )

        mock_attachment = ResolvedAttachment(
            file_id="file_img",
            title="photo.png",
            mime_type="image/png",
            content=b"\x89PNG...",
            source_url="https://drive.google.com/file/d/file_img/view",
            original_mime_type="image/png",
        )
        resolver = mock_resolver_cls.return_value
        resolver.resolve.return_value = iter([mock_attachment])

        mock_get_parser.return_value = None

        loader = _make_loader(
            course_ids=["12345"],
            load_attachments=True,
            parse_attachments=True,
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["content_type"] == "assignment"

    @patch(
        "langchain_google_classroom.drive_resolver.DriveAttachmentResolver",
        autospec=True,
    )
    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_custom_file_parser_cls(
        self,
        mock_fetcher_cls: MagicMock,
        mock_resolver_cls: MagicMock,
    ) -> None:
        """A custom file_parser_cls should be used for all attachments."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter(
            [SAMPLE_COURSE_WORK_WITH_ATTACHMENT]
        )
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])

        from langchain_google_classroom.drive_resolver import (
            ResolvedAttachment,
        )

        mock_attachment = ResolvedAttachment(
            file_id="file_aaa",
            title="report.pdf",
            mime_type="application/pdf",
            content=b"PDF bytes",
            source_url="https://drive.google.com/file/d/file_aaa/view",
            original_mime_type="application/pdf",
        )
        resolver = mock_resolver_cls.return_value
        resolver.resolve.return_value = iter([mock_attachment])

        # Custom parser class
        class _MockParser(BaseBlobParser):
            def lazy_parse(self, blob: Blob) -> Iterator[Document]:
                yield Document(
                    page_content="Custom parsed content",
                    metadata={"custom_key": "custom_value"},
                )

        loader = _make_loader(
            course_ids=["12345"],
            load_attachments=True,
            parse_attachments=True,
            file_parser_cls=_MockParser,
        )
        docs = loader.load()

        assert len(docs) == 2
        assert docs[1].metadata["content_type"] == "assignment_attachment"
        assert "Custom parsed content" in docs[1].page_content
        # Custom metadata is preserved alongside merged metadata
        assert docs[1].metadata["custom_key"] == "custom_value"
        assert docs[1].metadata["source"] == "google_classroom"

    @patch(
        "langchain_google_classroom.loader.get_parser",
    )
    @patch(
        "langchain_google_classroom.drive_resolver.DriveAttachmentResolver",
        autospec=True,
    )
    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_images_skipped_by_default(
        self,
        mock_fetcher_cls: MagicMock,
        mock_resolver_cls: MagicMock,
        mock_get_parser: MagicMock,
    ) -> None:
        """Image attachments should be skipped when load_images=False."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter(
            [SAMPLE_COURSE_WORK_WITH_ATTACHMENT]
        )
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])

        from langchain_google_classroom.drive_resolver import (
            ResolvedAttachment,
        )

        mock_attachment = ResolvedAttachment(
            file_id="file_img",
            title="photo.png",
            mime_type="image/png",
            content=b"\x89PNG...",
            source_url="https://drive.google.com/file/d/img/view",
            original_mime_type="image/png",
        )
        resolver = mock_resolver_cls.return_value
        resolver.resolve.return_value = iter([mock_attachment])

        loader = _make_loader(
            course_ids=["12345"],
            load_attachments=True,
            parse_attachments=True,
            load_images=False,
        )
        docs = loader.load()

        # Only the assignment doc, no image
        assert len(docs) == 1
        assert docs[0].metadata["content_type"] == "assignment"

    @patch(
        "langchain_google_classroom.drive_resolver.DriveAttachmentResolver",
        autospec=True,
    )
    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_images_enabled_with_vision(
        self,
        mock_fetcher_cls: MagicMock,
        mock_resolver_cls: MagicMock,
    ) -> None:
        """Image attachments should be parsed when load_images=True."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter(
            [SAMPLE_COURSE_WORK_WITH_ATTACHMENT]
        )
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])

        from langchain_google_classroom.drive_resolver import (
            ResolvedAttachment,
        )

        mock_attachment = ResolvedAttachment(
            file_id="file_img",
            title="chart.png",
            mime_type="image/png",
            content=b"\x89PNG fake image",
            source_url="https://drive.google.com/file/d/img/view",
            original_mime_type="image/png",
        )
        resolver = mock_resolver_cls.return_value
        resolver.resolve.return_value = iter([mock_attachment])

        mock_vision = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "A chart showing student scores."
        mock_vision.invoke.return_value = mock_response

        loader = _make_loader(
            course_ids=["12345"],
            load_attachments=True,
            parse_attachments=True,
            load_images=True,
            vision_model=mock_vision,
        )
        docs = loader.load()

        assert len(docs) == 2
        assert docs[1].metadata["content_type"] == "assignment_attachment"
        assert "A chart showing student scores." in docs[1].page_content


# ---------------------------------------------------------------------------
# Submissions integration
# ---------------------------------------------------------------------------


class TestLoaderSubmissions:
    """Verify submissions are yielded from lazy_load."""

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_submissions_loaded_when_enabled(self, mock_fetcher_cls: MagicMock) -> None:
        """Submissions should appear when load_submissions=True."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter([])
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])
        fetcher.list_student_submissions.return_value = iter(
            [
                {
                    "id": "sub_001",
                    "courseWorkId": "cw_001",
                    "userId": "u1",
                    "state": "TURNED_IN",
                },
            ]
        )

        loader = _make_loader(
            course_ids=["12345"],
            load_attachments=False,
            load_assignments=False,
            load_announcements=False,
            load_materials=False,
            load_submissions=True,
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["content_type"] == "submission"
        fetcher.list_student_submissions.assert_called_once_with("12345")

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_submissions_not_loaded_when_disabled(
        self, mock_fetcher_cls: MagicMock
    ) -> None:
        """Submissions API should not be called when disabled."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter([])
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])

        loader = _make_loader(
            course_ids=["12345"],
            load_attachments=False,
            load_submissions=False,
        )
        loader.load()
        fetcher.list_student_submissions.assert_not_called()


# ---------------------------------------------------------------------------
# Topics integration
# ---------------------------------------------------------------------------


class TestLoaderTopics:
    """Verify topics are yielded from lazy_load."""

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_topics_loaded_when_enabled(self, mock_fetcher_cls: MagicMock) -> None:
        """Topics should appear when load_topics=True."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter([])
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])
        fetcher.list_topics.return_value = iter(
            [
                {"topicId": "t1", "name": "Week 1"},
            ]
        )

        loader = _make_loader(
            course_ids=["12345"],
            load_attachments=False,
            load_assignments=False,
            load_announcements=False,
            load_materials=False,
            load_topics=True,
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["content_type"] == "topic"

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_topics_not_loaded_when_disabled(self, mock_fetcher_cls: MagicMock) -> None:
        """Topics API should not be called when disabled."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter([])
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])

        loader = _make_loader(
            course_ids=["12345"],
            load_attachments=False,
            load_topics=False,
        )
        loader.load()
        fetcher.list_topics.assert_not_called()


# ---------------------------------------------------------------------------
# Roster integration
# ---------------------------------------------------------------------------


class TestLoaderRoster:
    """Verify roster data is yielded from lazy_load."""

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_roster_loaded_when_enabled(self, mock_fetcher_cls: MagicMock) -> None:
        """Students and teachers should appear when load_roster=True."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter([])
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])
        fetcher.list_students.return_value = iter(
            [
                {
                    "userId": "u1",
                    "profile": {
                        "name": {"fullName": "Alice"},
                        "emailAddress": "alice@test.com",
                    },
                },
            ]
        )
        fetcher.list_teachers.return_value = iter(
            [
                {
                    "userId": "t1",
                    "profile": {
                        "name": {"fullName": "Prof. Smith"},
                        "emailAddress": "smith@test.com",
                    },
                },
            ]
        )

        loader = _make_loader(
            course_ids=["12345"],
            load_attachments=False,
            load_assignments=False,
            load_announcements=False,
            load_materials=False,
            load_roster=True,
        )
        docs = loader.load()

        assert len(docs) == 2
        assert docs[0].metadata["content_type"] == "student"
        assert docs[1].metadata["content_type"] == "teacher"

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_roster_not_loaded_when_disabled(self, mock_fetcher_cls: MagicMock) -> None:
        """Roster API should not be called when disabled."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter([])
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])

        loader = _make_loader(
            course_ids=["12345"],
            load_attachments=False,
            load_roster=False,
        )
        loader.load()
        fetcher.list_students.assert_not_called()
        fetcher.list_teachers.assert_not_called()


# ---------------------------------------------------------------------------
# Rubrics integration
# ---------------------------------------------------------------------------


class TestLoaderRubrics:
    """Verify rubrics are fetched inline with courseWork."""

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_rubrics_yielded_after_assignment(
        self, mock_fetcher_cls: MagicMock
    ) -> None:
        """Each courseWork item should have its rubrics fetched inline."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter(
            [
                {
                    "id": "cw_001",
                    "title": "Homework 1",
                    "description": "Do the exercises",
                    "state": "PUBLISHED",
                    "creationTime": "2024-01-10T00:00:00Z",
                    "updateTime": "2024-01-10T00:00:00Z",
                    "alternateLink": "https://example.com",
                },
            ]
        )
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])
        fetcher.list_rubrics.return_value = iter(
            [
                {
                    "id": "r_001",
                    "courseWorkId": "cw_001",
                    "criteria": [
                        {
                            "title": "Completeness",
                            "levels": [
                                {"title": "Full", "points": 10},
                            ],
                        }
                    ],
                },
            ]
        )

        loader = _make_loader(
            course_ids=["12345"],
            load_attachments=False,
            load_assignments=True,
            load_announcements=False,
            load_materials=False,
        )
        docs = loader.load()

        assert len(docs) == 2
        assert docs[0].metadata["content_type"] == "assignment"
        assert docs[1].metadata["content_type"] == "rubric"
        assert docs[1].metadata["course_work_title"] == "Homework 1"
        assert "Completeness" in docs[1].page_content
        fetcher.list_rubrics.assert_called_once_with("12345", "cw_001")

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_no_rubrics_when_assignments_disabled(
        self, mock_fetcher_cls: MagicMock
    ) -> None:
        """Rubrics should not be fetched if assignments are disabled."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])

        loader = _make_loader(
            course_ids=["12345"],
            load_attachments=False,
            load_assignments=False,
            load_announcements=False,
            load_materials=False,
        )
        loader.load()
        fetcher.list_rubrics.assert_not_called()


# ---------------------------------------------------------------------------
# All features combined
# ---------------------------------------------------------------------------


class TestLoaderAllFeatures:
    """Verify all content types can be loaded together."""

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_all_content_types(self, mock_fetcher_cls: MagicMock) -> None:
        """All 7 content types should be present when all flags enabled."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter(
            [
                {
                    "id": "cw1",
                    "title": "HW1",
                    "state": "PUBLISHED",
                    "creationTime": "",
                    "updateTime": "",
                    "alternateLink": "",
                },
            ]
        )
        fetcher.list_rubrics.return_value = iter([])
        fetcher.list_announcements.return_value = iter(
            [
                {
                    "id": "ann1",
                    "text": "Hello",
                    "state": "PUBLISHED",
                    "creationTime": "",
                    "updateTime": "",
                    "alternateLink": "",
                },
            ]
        )
        fetcher.list_course_work_materials.return_value = iter(
            [
                {
                    "id": "mat1",
                    "title": "Slides",
                    "state": "PUBLISHED",
                    "creationTime": "",
                    "updateTime": "",
                    "alternateLink": "",
                },
            ]
        )
        fetcher.list_student_submissions.return_value = iter(
            [
                {"id": "sub1", "courseWorkId": "cw1", "state": "TURNED_IN"},
            ]
        )
        fetcher.list_topics.return_value = iter(
            [
                {"topicId": "t1", "name": "Week 1"},
            ]
        )
        fetcher.list_students.return_value = iter(
            [
                {"userId": "u1", "profile": {"name": {"fullName": "Alice"}}},
            ]
        )
        fetcher.list_teachers.return_value = iter(
            [
                {"userId": "t1", "profile": {"name": {"fullName": "Prof"}}},
            ]
        )

        loader = _make_loader(
            course_ids=["12345"],
            load_attachments=False,
            load_assignments=True,
            load_announcements=True,
            load_materials=True,
            load_submissions=True,
            load_topics=True,
            load_roster=True,
        )
        docs = loader.load()

        content_types = [d.metadata["content_type"] for d in docs]
        assert "assignment" in content_types
        assert "announcement" in content_types
        assert "material" in content_types
        assert "submission" in content_types
        assert "topic" in content_types
        assert "student" in content_types
        assert "teacher" in content_types
        assert len(docs) == 7
