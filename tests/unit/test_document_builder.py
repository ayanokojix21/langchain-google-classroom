"""Unit tests for the document builder functions."""

from __future__ import annotations

from typing import Any, Dict

import pytest
from langchain_core.documents import Document

from langchain_google_classroom.document_builder import (
    _format_due_date,
    build_course_meta,
    build_from_announcement,
    build_from_course_work,
    build_from_material,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COURSE_META: Dict[str, Any] = {
    "course_id": "12345",
    "course_name": "Machine Learning",
}


# ---------------------------------------------------------------------------
# Tests — build_course_meta
# ---------------------------------------------------------------------------


class TestBuildCourseMeta:
    """Tests for build_course_meta."""

    def test_extracts_fields(self) -> None:
        course = {"id": "c1", "name": "Math 101", "extra": "ignored"}
        meta = build_course_meta(course)
        assert meta == {"course_id": "c1", "course_name": "Math 101"}

    def test_missing_fields(self) -> None:
        meta = build_course_meta({})
        assert meta == {"course_id": "", "course_name": ""}


# ---------------------------------------------------------------------------
# Tests — _format_due_date
# ---------------------------------------------------------------------------


class TestFormatDueDate:
    """Tests for _format_due_date."""

    def test_full_due_date_with_time(self) -> None:
        item = {
            "dueDate": {"year": 2024, "month": 3, "day": 15},
            "dueTime": {"hours": 23, "minutes": 59},
        }
        assert _format_due_date(item) == "2024-03-15T23:59:00"

    def test_due_date_without_time(self) -> None:
        item = {"dueDate": {"year": 2024, "month": 1, "day": 1}}
        assert _format_due_date(item) == "2024-01-01"

    def test_no_due_date(self) -> None:
        assert _format_due_date({}) is None

    def test_due_date_with_zero_padded_values(self) -> None:
        item = {
            "dueDate": {"year": 2024, "month": 2, "day": 5},
            "dueTime": {"hours": 9, "minutes": 0},
        }
        assert _format_due_date(item) == "2024-02-05T09:00:00"


# ---------------------------------------------------------------------------
# Tests — build_from_course_work
# ---------------------------------------------------------------------------


class TestBuildFromCourseWork:
    """Tests for build_from_course_work."""

    def test_basic_assignment(self) -> None:
        item = {
            "id": "cw1",
            "title": "Homework 1",
            "description": "Complete exercises.",
            "state": "PUBLISHED",
            "creationTime": "2024-01-10T08:00:00Z",
            "updateTime": "2024-01-10T09:00:00Z",
            "alternateLink": "https://classroom.google.com/test",
            "maxPoints": 50.0,
            "dueDate": {"year": 2024, "month": 1, "day": 20},
            "dueTime": {"hours": 23, "minutes": 59},
        }
        doc = build_from_course_work(item, COURSE_META)

        assert isinstance(doc, Document)
        assert "Assignment: Homework 1" in doc.page_content
        assert "Complete exercises." in doc.page_content
        assert doc.metadata["source"] == "google_classroom"
        assert doc.metadata["course_id"] == "12345"
        assert doc.metadata["content_type"] == "assignment"
        assert doc.metadata["title"] == "Homework 1"
        assert doc.metadata["item_id"] == "cw1"
        assert doc.metadata["due_date"] == "2024-01-20T23:59:00"
        assert doc.metadata["max_points"] == 50.0

    def test_assignment_without_description(self) -> None:
        item = {"id": "cw2", "title": "Quiz"}
        doc = build_from_course_work(item, COURSE_META)
        assert "Assignment: Quiz" in doc.page_content
        # No blank line / description after title
        assert doc.page_content.strip() == "Assignment: Quiz"

    def test_assignment_without_due_date(self) -> None:
        item = {"id": "cw3", "title": "Extra Credit"}
        doc = build_from_course_work(item, COURSE_META)
        assert "due_date" not in doc.metadata


# ---------------------------------------------------------------------------
# Tests — build_from_announcement
# ---------------------------------------------------------------------------


class TestBuildFromAnnouncement:
    """Tests for build_from_announcement."""

    def test_basic_announcement(self) -> None:
        item = {
            "id": "ann1",
            "text": "Welcome to the course!",
            "state": "PUBLISHED",
            "creationTime": "2024-01-01T10:00:00Z",
            "updateTime": "2024-01-01T10:00:00Z",
            "alternateLink": "https://classroom.google.com/ann",
        }
        doc = build_from_announcement(item, COURSE_META)

        assert isinstance(doc, Document)
        assert "Announcement: Welcome to the course!" in doc.page_content
        assert doc.metadata["content_type"] == "announcement"
        assert doc.metadata["title"] == "Welcome to the course!"

    def test_long_announcement_title_truncated(self) -> None:
        long_text = "A" * 200
        item = {"id": "ann2", "text": long_text}
        doc = build_from_announcement(item, COURSE_META)
        assert len(doc.metadata["title"]) == 80

    def test_empty_announcement(self) -> None:
        item = {"id": "ann3", "text": ""}
        doc = build_from_announcement(item, COURSE_META)
        assert doc.metadata["title"] == "Untitled Announcement"


# ---------------------------------------------------------------------------
# Tests — build_from_material
# ---------------------------------------------------------------------------


class TestBuildFromMaterial:
    """Tests for build_from_material."""

    def test_basic_material(self) -> None:
        item = {
            "id": "mat1",
            "title": "Lecture Notes",
            "description": "Week 1 notes on linear algebra.",
            "state": "PUBLISHED",
            "creationTime": "2024-01-08T09:00:00Z",
            "updateTime": "2024-01-08T09:00:00Z",
            "alternateLink": "https://classroom.google.com/mat",
        }
        doc = build_from_material(item, COURSE_META)

        assert isinstance(doc, Document)
        assert "Material: Lecture Notes" in doc.page_content
        assert "Week 1 notes on linear algebra." in doc.page_content
        assert doc.metadata["content_type"] == "material"

    def test_material_without_description(self) -> None:
        item = {"id": "mat2", "title": "Slides"}
        doc = build_from_material(item, COURSE_META)
        assert doc.page_content.strip() == "Material: Slides"
