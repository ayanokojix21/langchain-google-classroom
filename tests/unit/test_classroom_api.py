"""Unit tests for ClassroomAPIFetcher."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from langchain_google_classroom.classroom_api import ClassroomAPIFetcher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fetcher() -> ClassroomAPIFetcher:
    """Create a fetcher with a mocked Google API service."""
    with patch(
        "langchain_google_classroom.classroom_api._import_googleapiclient_build"
    ) as mock_build_fn:
        mock_service = MagicMock()
        mock_build_fn.return_value = MagicMock(return_value=mock_service)
        fetcher = ClassroomAPIFetcher(credentials=MagicMock())
        # Store mock service for test assertions
        fetcher._mock_service = mock_service  # type: ignore[attr-defined]
    return fetcher


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassroomAPIFetcher:
    """Tests for ClassroomAPIFetcher."""

    def test_list_courses_by_id(self) -> None:
        """When course_ids are provided, each is fetched via courses.get."""
        fetcher = _make_fetcher()
        course_data = {"id": "123", "name": "Test Course"}
        fetcher._mock_service.courses().get().execute.return_value = (  # type: ignore[attr-defined]
            course_data
        )

        courses = list(fetcher.list_courses(course_ids=["123"]))

        assert len(courses) == 1
        assert courses[0]["id"] == "123"

    def test_list_courses_all(self) -> None:
        """When no course_ids, paginate via courses.list."""
        fetcher = _make_fetcher()
        mock_courses = fetcher._mock_service.courses()  # type: ignore[attr-defined]

        # Simulate a single page of results
        mock_courses.list.return_value.execute.return_value = {
            "courses": [
                {"id": "a", "name": "Course A"},
                {"id": "b", "name": "Course B"},
            ]
        }
        mock_courses.list_next.return_value = None

        courses = list(fetcher.list_courses())

        assert len(courses) == 2
        assert courses[0]["id"] == "a"
        assert courses[1]["id"] == "b"

    def test_list_course_work_pagination(self) -> None:
        """Two pages of courseWork should both be yielded."""
        fetcher = _make_fetcher()
        mock_cw = fetcher._mock_service.courses().courseWork()  # type: ignore[attr-defined]

        # Page 1
        page1_response = {
            "courseWork": [
                {"id": "cw1", "title": "HW 1"},
                {"id": "cw2", "title": "HW 2"},
            ]
        }
        # Page 2
        page2_response = {
            "courseWork": [
                {"id": "cw3", "title": "HW 3"},
            ]
        }

        # First call to list().execute() returns page 1
        mock_request_1 = MagicMock()
        mock_request_1.execute.return_value = page1_response
        mock_cw.list.return_value = mock_request_1

        # list_next returns a second request for page 2, then None
        mock_request_2 = MagicMock()
        mock_request_2.execute.return_value = page2_response
        mock_cw.list_next.side_effect = [mock_request_2, None]

        items = list(fetcher.list_course_work("123"))

        assert len(items) == 3
        assert [i["id"] for i in items] == ["cw1", "cw2", "cw3"]

    def test_list_course_work_error_handling(self) -> None:
        """Errors during courseWork fetch should be logged, not raised."""
        fetcher = _make_fetcher()
        mock_cw = fetcher._mock_service.courses().courseWork()  # type: ignore[attr-defined]
        mock_cw.list.return_value.execute.side_effect = Exception(
            "403 Forbidden"
        )

        # Should not raise
        items = list(fetcher.list_course_work("999"))
        assert items == []

    def test_list_announcements(self) -> None:
        """Announcements should be yielded from API response."""
        fetcher = _make_fetcher()
        mock_ann = fetcher._mock_service.courses().announcements()  # type: ignore[attr-defined]

        mock_ann.list.return_value.execute.return_value = {
            "announcements": [
                {"id": "a1", "text": "Hello"},
            ]
        }
        mock_ann.list_next.return_value = None

        items = list(fetcher.list_announcements("123"))

        assert len(items) == 1
        assert items[0]["text"] == "Hello"

    def test_list_course_work_materials(self) -> None:
        """Materials should be yielded from API response."""
        fetcher = _make_fetcher()
        mock_mat = fetcher._mock_service.courses().courseWorkMaterials()  # type: ignore[attr-defined]

        mock_mat.list.return_value.execute.return_value = {
            "courseWorkMaterial": [
                {"id": "m1", "title": "Slides"},
            ]
        }
        mock_mat.list_next.return_value = None

        items = list(fetcher.list_course_work_materials("123"))

        assert len(items) == 1
        assert items[0]["title"] == "Slides"

    def test_list_courses_error_skipped(self) -> None:
        """A failing courses.get should log and skip, not crash."""
        fetcher = _make_fetcher()
        fetcher._mock_service.courses().get().execute.side_effect = (  # type: ignore[attr-defined]
            Exception("Not found")
        )

        courses = list(fetcher.list_courses(course_ids=["bad_id"]))
        assert courses == []
