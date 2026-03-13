"""Unit tests for GoogleClassroomLoader."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

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
# Tests
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

        loader = _make_loader(course_ids=["12345"])
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
        # No due_date on second item
        assert "due_date" not in docs[1].metadata

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_lazy_load_announcements(
        self, mock_fetcher_cls: MagicMock
    ) -> None:
        """One announcement should produce one Document."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([SAMPLE_COURSE])
        fetcher.list_course_work.return_value = iter([])
        fetcher.list_announcements.return_value = iter([SAMPLE_ANNOUNCEMENT])
        fetcher.list_course_work_materials.return_value = iter([])

        loader = _make_loader(course_ids=["12345"])
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
        fetcher.list_course_work_materials.return_value = iter(
            [SAMPLE_MATERIAL]
        )

        loader = _make_loader(course_ids=["12345"])
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
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["content_type"] == "assignment"
        # list_announcements should never be called
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

        loader = _make_loader(course_ids=["12345"])
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

        loader = _make_loader()
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

        loader = _make_loader(course_ids=["aaa", "bbb"])
        docs = loader.load()

        assert len(docs) == 2
        assert docs[0].metadata["course_name"] == "Course A"
        assert docs[1].metadata["course_name"] == "Course B"

    def test_credentials_passthrough(self) -> None:
        """Pre-built credentials should be used directly without calling
        ``get_classroom_credentials``."""
        creds = MagicMock()
        loader = GoogleClassroomLoader(credentials=creds)
        assert loader._get_credentials() is creds
