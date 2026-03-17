"""Unit tests for ClassroomAPIFetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_google_classroom.classroom_api import ClassroomAPIFetcher

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fetcher() -> tuple[ClassroomAPIFetcher, MagicMock]:
    """Create a fetcher with a mocked Google API service."""
    with patch(
        "langchain_google_classroom.classroom_api._import_googleapiclient_build"
    ) as mock_build_fn:
        mock_service = MagicMock()
        mock_build_fn.return_value = MagicMock(return_value=mock_service)
        fetcher = ClassroomAPIFetcher(credentials=MagicMock())
    return fetcher, mock_service


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassroomAPIFetcher:
    """Tests for ClassroomAPIFetcher."""

    def test_list_courses_by_id(
        self,
        fetcher: tuple[ClassroomAPIFetcher, MagicMock],
    ) -> None:
        """When course_ids are provided, each is fetched via courses.get."""
        fetcher_obj, mock_service = fetcher
        course_data = {"id": "123", "name": "Test Course"}
        mock_courses = mock_service.courses.return_value
        mock_courses.get.return_value.execute.return_value = course_data

        courses = list(fetcher_obj.list_courses(course_ids=["123"]))

        assert courses == [course_data]
        mock_courses.get.assert_called_once_with(id="123")

    def test_list_courses_all(
        self,
        fetcher: tuple[ClassroomAPIFetcher, MagicMock],
    ) -> None:
        """When no course_ids, paginate via courses.list."""
        fetcher_obj, mock_service = fetcher
        mock_courses = mock_service.courses.return_value

        # Simulate a single page of results
        mock_courses.list.return_value.execute.return_value = {
            "courses": [
                {"id": "a", "name": "Course A"},
                {"id": "b", "name": "Course B"},
            ]
        }
        mock_courses.list_next.return_value = None

        courses = list(fetcher_obj.list_courses())

        assert [c["id"] for c in courses] == ["a", "b"]
        mock_courses.list.assert_called_once_with(pageSize=100)
        mock_courses.list_next.assert_called_once()

    def test_list_course_work_pagination(
        self,
        fetcher: tuple[ClassroomAPIFetcher, MagicMock],
    ) -> None:
        """Two pages of courseWork should both be yielded."""
        fetcher_obj, mock_service = fetcher
        mock_cw = mock_service.courses.return_value.courseWork.return_value

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

        items = list(fetcher_obj.list_course_work("123"))

        assert [i["id"] for i in items] == ["cw1", "cw2", "cw3"]
        mock_cw.list.assert_called_once_with(courseId="123", pageSize=100)

    def test_list_course_work_error_handling(
        self,
        fetcher: tuple[ClassroomAPIFetcher, MagicMock],
    ) -> None:
        """Errors during courseWork fetch should be logged, not raised."""
        fetcher_obj, mock_service = fetcher
        mock_cw = mock_service.courses.return_value.courseWork.return_value
        mock_cw.list.return_value.execute.side_effect = Exception("403 Forbidden")

        # Should not raise
        items = list(fetcher_obj.list_course_work("999"))
        assert items == []

    def test_list_announcements(
        self,
        fetcher: tuple[ClassroomAPIFetcher, MagicMock],
    ) -> None:
        """Announcements should be yielded from API response."""
        fetcher_obj, mock_service = fetcher
        mock_ann = mock_service.courses.return_value.announcements.return_value

        mock_ann.list.return_value.execute.return_value = {
            "announcements": [
                {"id": "a1", "text": "Hello"},
            ]
        }
        mock_ann.list_next.return_value = None

        items = list(fetcher_obj.list_announcements("123"))

        assert items == [{"id": "a1", "text": "Hello"}]
        mock_ann.list.assert_called_once_with(courseId="123", pageSize=100)

    def test_list_course_work_materials(
        self,
        fetcher: tuple[ClassroomAPIFetcher, MagicMock],
    ) -> None:
        """Materials should be yielded from API response."""
        fetcher_obj, mock_service = fetcher
        mock_mat = mock_service.courses.return_value.courseWorkMaterials.return_value

        mock_mat.list.return_value.execute.return_value = {
            "courseWorkMaterial": [
                {"id": "m1", "title": "Slides"},
            ]
        }
        mock_mat.list_next.return_value = None

        items = list(fetcher_obj.list_course_work_materials("123"))

        assert items == [{"id": "m1", "title": "Slides"}]
        mock_mat.list.assert_called_once_with(courseId="123", pageSize=100)

    def test_list_courses_error_skipped(
        self,
        fetcher: tuple[ClassroomAPIFetcher, MagicMock],
    ) -> None:
        """A failing courses.get should log and skip, not crash."""
        fetcher_obj, mock_service = fetcher
        mock_courses = mock_service.courses.return_value
        mock_courses.get.return_value.execute.side_effect = Exception("Not found")

        courses = list(fetcher_obj.list_courses(course_ids=["bad_id"]))
        assert courses == []
        mock_courses.get.assert_called_once_with(id="bad_id")
