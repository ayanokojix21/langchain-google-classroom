"""Unit tests for the document builder functions."""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.documents import Document

from langchain_google_classroom.document_builder import (
    _extract_profile_name,
    _format_due_date,
    _format_rubric_criteria,
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

    def test_assignment_title_fallback(self) -> None:
        item = {"id": "cw4"}
        doc = build_from_course_work(item, COURSE_META)
        assert doc.metadata["title"] == "Untitled Assignment"
        assert doc.page_content.strip() == "Assignment: Untitled Assignment"

    def test_assignment_content_normalized(self) -> None:
        item = {
            "id": "cw5",
            "title": "Normalization Test",
            "description": "Line 1\r\n\r\n\r\nLine 2\x00",
        }
        doc = build_from_course_work(item, COURSE_META)
        assert "\r" not in doc.page_content
        assert "\x00" not in doc.page_content
        assert "\n\n\n" not in doc.page_content


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

    def test_material_title_fallback(self) -> None:
        item = {"id": "mat3"}
        doc = build_from_material(item, COURSE_META)
        assert doc.metadata["title"] == "Untitled Material"
        assert doc.page_content.strip() == "Material: Untitled Material"


# ---------------------------------------------------------------------------
# Tests — build_from_attachment
# ---------------------------------------------------------------------------


class TestBuildFromAttachment:
    """Tests for build_from_attachment."""

    def test_basic_attachment(self) -> None:
        parent_item = {
            "id": "cw_attach_1",
            "title": "Homework With File",
            "creationTime": "2024-02-01T10:00:00Z",
            "updateTime": "2024-02-01T11:00:00Z",
            "alternateLink": "https://classroom.google.com/attach",
        }

        doc = build_from_attachment(
            file_id="file123",
            title="instructions.pdf",
            mime_type="application/pdf",
            source_url="https://drive.google.com/file/d/file123/view",
            original_mime_type="application/pdf",
            parsed_text="Read the attached instructions.",
            parent_item=parent_item,
            course_meta=COURSE_META,
            content_type="assignment",
        )

        assert isinstance(doc, Document)
        assert doc.page_content == "Read the attached instructions."
        assert doc.metadata["content_type"] == "assignment_attachment"
        assert doc.metadata["title"] == "instructions.pdf"
        assert doc.metadata["file_id"] == "file123"
        assert doc.metadata["parent_title"] == "Homework With File"
        assert doc.metadata["mime_type"] == "application/pdf"
        assert (
            doc.metadata["attachment_url"]
            == "https://drive.google.com/file/d/file123/view"
        )

    def test_attachment_parent_title_fallback_from_text(self) -> None:
        parent_item = {
            "id": "ann_attach_1",
            "text": "Important announcement with a linked file",
        }

        doc = build_from_attachment(
            file_id="file456",
            title="notes.txt",
            mime_type="text/plain",
            source_url="https://drive.google.com/file/d/file456/view",
            original_mime_type="text/plain",
            parsed_text="Raw\r\n\r\n\r\nText\x00",
            parent_item=parent_item,
            course_meta=COURSE_META,
            content_type="announcement",
        )

        assert (
            doc.metadata["parent_title"] == "Important announcement with a linked file"
        )
        assert doc.metadata["content_type"] == "announcement_attachment"
        assert "\r" not in doc.page_content
        assert "\x00" not in doc.page_content
        assert "\n\n\n" not in doc.page_content


# ---------------------------------------------------------------------------
# Tests — build_from_submission
# ---------------------------------------------------------------------------


class TestBuildFromSubmission:
    """Tests for build_from_submission."""

    def test_basic_submission(self) -> None:
        item = {
            "id": "sub_001",
            "courseWorkId": "cw_001",
            "userId": "user_123",
            "state": "TURNED_IN",
            "creationTime": "2024-01-10T10:00:00Z",
            "updateTime": "2024-01-10T12:00:00Z",
            "alternateLink": "https://classroom.google.com/sub",
        }
        doc = build_from_submission(item, COURSE_META)

        assert isinstance(doc, Document)
        assert doc.metadata["content_type"] == "submission"
        assert doc.metadata["item_id"] == "sub_001"
        assert doc.metadata["course_work_id"] == "cw_001"
        assert doc.metadata["user_id"] == "user_123"
        assert doc.metadata["state"] == "TURNED_IN"
        assert doc.metadata["late"] is False
        assert "Student Submission" in doc.page_content
        assert "State: TURNED_IN" in doc.page_content

    def test_late_submission(self) -> None:
        item = {
            "id": "sub_002",
            "courseWorkId": "cw_001",
            "state": "TURNED_IN",
            "late": True,
        }
        doc = build_from_submission(item, COURSE_META)
        assert doc.metadata["late"] is True
        assert "Late: Yes" in doc.page_content

    def test_graded_submission(self) -> None:
        item = {
            "id": "sub_003",
            "courseWorkId": "cw_001",
            "state": "RETURNED",
            "assignedGrade": 95,
            "draftGrade": 90,
        }
        doc = build_from_submission(item, COURSE_META)
        assert doc.metadata["assigned_grade"] == 95.0
        assert doc.metadata["draft_grade"] == 90.0
        assert "Grade: 95" in doc.page_content

    def test_short_answer_submission(self) -> None:
        item = {
            "id": "sub_004",
            "courseWorkId": "cw_001",
            "state": "TURNED_IN",
            "shortAnswerSubmission": {"answer": "42"},
        }
        doc = build_from_submission(item, COURSE_META)
        assert "Answer: 42" in doc.page_content

    def test_multiple_choice_submission(self) -> None:
        item = {
            "id": "sub_005",
            "courseWorkId": "cw_001",
            "state": "TURNED_IN",
            "multipleChoiceSubmission": {"answer": "Option B"},
        }
        doc = build_from_submission(item, COURSE_META)
        assert "Answer: Option B" in doc.page_content

    def test_assignment_submission_with_attachments(self) -> None:
        item = {
            "id": "sub_006",
            "courseWorkId": "cw_001",
            "state": "TURNED_IN",
            "assignmentSubmission": {
                "attachments": [
                    {"driveFile": {"title": "essay.pdf"}},
                    {"driveFile": {"title": "data.csv"}},
                ]
            },
        }
        doc = build_from_submission(item, COURSE_META)
        assert "Attachments: essay.pdf, data.csv" in doc.page_content


# ---------------------------------------------------------------------------
# Tests — build_from_rubric
# ---------------------------------------------------------------------------


class TestBuildFromRubric:
    """Tests for build_from_rubric."""

    def test_basic_rubric(self) -> None:
        item = {
            "id": "r_001",
            "courseWorkId": "cw_001",
            "creationTime": "2024-01-10T10:00:00Z",
            "updateTime": "2024-01-10T12:00:00Z",
            "criteria": [
                {
                    "title": "Content Quality",
                    "description": "Evaluate the content",
                    "levels": [
                        {
                            "title": "Excellent",
                            "points": 10,
                            "description": "Outstanding work",
                        },
                        {
                            "title": "Good",
                            "points": 7,
                            "description": "Solid work",
                        },
                        {
                            "title": "Poor",
                            "points": 3,
                            "description": "Needs improvement",
                        },
                    ],
                }
            ],
        }
        doc = build_from_rubric(item, COURSE_META, course_work_title="Homework 1")

        assert isinstance(doc, Document)
        assert doc.metadata["content_type"] == "rubric"
        assert doc.metadata["item_id"] == "r_001"
        assert doc.metadata["course_work_id"] == "cw_001"
        assert doc.metadata["criteria_count"] == 1
        assert doc.metadata["course_work_title"] == "Homework 1"
        assert "Rubric for: Homework 1" in doc.page_content
        assert "Content Quality" in doc.page_content
        assert "Excellent" in doc.page_content
        assert "10 pts" in doc.page_content

    def test_rubric_without_title(self) -> None:
        item = {"id": "r_002", "courseWorkId": "cw_002", "criteria": []}
        doc = build_from_rubric(item, COURSE_META)
        assert "Rubric for: cw_002" in doc.page_content
        assert "course_work_title" not in doc.metadata

    def test_rubric_multiple_criteria(self) -> None:
        item = {
            "id": "r_003",
            "courseWorkId": "cw_003",
            "criteria": [
                {"title": "Grammar", "levels": []},
                {"title": "Creativity", "levels": []},
                {"title": "Research", "levels": []},
            ],
        }
        doc = build_from_rubric(item, COURSE_META)
        assert doc.metadata["criteria_count"] == 3
        assert "Grammar" in doc.page_content
        assert "Creativity" in doc.page_content
        assert "Research" in doc.page_content


# ---------------------------------------------------------------------------
# Tests — _format_rubric_criteria
# ---------------------------------------------------------------------------


class TestFormatRubricCriteria:
    """Tests for _format_rubric_criteria helper."""

    def test_criterion_with_levels(self) -> None:
        criteria = [
            {
                "title": "Analysis",
                "description": "Depth of analysis",
                "levels": [
                    {
                        "title": "Excellent",
                        "points": 10,
                        "description": "Deep insight",
                    },
                    {"title": "Fair", "points": 5},
                ],
            }
        ]
        result = _format_rubric_criteria(criteria)
        assert "Criterion 1: Analysis" in result
        assert "Description: Depth of analysis" in result
        assert "Excellent (10 pts): Deep insight" in result
        assert "Fair (5 pts)" in result

    def test_empty_criteria(self) -> None:
        assert _format_rubric_criteria([]) == ""

    def test_unscored_levels(self) -> None:
        criteria = [
            {
                "title": "Participation",
                "levels": [{"title": "Active"}, {"title": "Passive"}],
            }
        ]
        result = _format_rubric_criteria(criteria)
        assert "Active" in result
        assert "pts" not in result


# ---------------------------------------------------------------------------
# Tests — build_from_topic
# ---------------------------------------------------------------------------


class TestBuildFromTopic:
    """Tests for build_from_topic."""

    def test_basic_topic(self) -> None:
        item = {
            "topicId": "topic_001",
            "name": "Linear Algebra",
            "updateTime": "2024-01-10T10:00:00Z",
        }
        doc = build_from_topic(item, COURSE_META)

        assert isinstance(doc, Document)
        assert doc.metadata["content_type"] == "topic"
        assert doc.metadata["title"] == "Linear Algebra"
        assert doc.metadata["item_id"] == "topic_001"
        assert "Topic: Linear Algebra" in doc.page_content

    def test_topic_without_name(self) -> None:
        item = {"topicId": "topic_002"}
        doc = build_from_topic(item, COURSE_META)
        assert doc.metadata["title"] == "Untitled Topic"
        assert "Topic: Untitled Topic" in doc.page_content


# ---------------------------------------------------------------------------
# Tests — _extract_profile_name
# ---------------------------------------------------------------------------


class TestExtractProfileName:
    """Tests for _extract_profile_name helper."""

    def test_full_name(self) -> None:
        profile = {"name": {"fullName": "Alice Johnson"}}
        assert _extract_profile_name(profile) == "Alice Johnson"

    def test_given_family_name(self) -> None:
        profile = {"name": {"givenName": "Bob", "familyName": "Smith"}}
        assert _extract_profile_name(profile) == "Bob Smith"

    def test_given_name_only(self) -> None:
        profile = {"name": {"givenName": "Charlie"}}
        assert _extract_profile_name(profile) == "Charlie"

    def test_empty_profile(self) -> None:
        assert _extract_profile_name({}) == "Unknown"

    def test_full_name_takes_priority(self) -> None:
        profile = {
            "name": {
                "fullName": "Full Name",
                "givenName": "Given",
                "familyName": "Family",
            }
        }
        assert _extract_profile_name(profile) == "Full Name"


# ---------------------------------------------------------------------------
# Tests — build_from_student
# ---------------------------------------------------------------------------


class TestBuildFromStudent:
    """Tests for build_from_student."""

    def test_basic_student(self) -> None:
        item = {
            "userId": "user_001",
            "profile": {
                "name": {"fullName": "Alice Johnson"},
                "emailAddress": "alice@school.edu",
            },
        }
        doc = build_from_student(item, COURSE_META)

        assert isinstance(doc, Document)
        assert doc.metadata["content_type"] == "student"
        assert doc.metadata["title"] == "Alice Johnson"
        assert doc.metadata["user_id"] == "user_001"
        assert doc.metadata["email"] == "alice@school.edu"
        assert "Student: Alice Johnson" in doc.page_content
        assert "Email: alice@school.edu" in doc.page_content

    def test_student_without_email(self) -> None:
        item = {
            "userId": "user_002",
            "profile": {"name": {"fullName": "Bob Smith"}},
        }
        doc = build_from_student(item, COURSE_META)
        assert "Email:" not in doc.page_content
        assert doc.metadata["email"] == ""

    def test_student_empty_profile(self) -> None:
        item = {"userId": "user_003"}
        doc = build_from_student(item, COURSE_META)
        assert doc.metadata["title"] == "Unknown"


# ---------------------------------------------------------------------------
# Tests — build_from_teacher
# ---------------------------------------------------------------------------


class TestBuildFromTeacher:
    """Tests for build_from_teacher."""

    def test_basic_teacher(self) -> None:
        item = {
            "userId": "teacher_001",
            "profile": {
                "name": {"fullName": "Prof. Smith"},
                "emailAddress": "smith@university.edu",
            },
        }
        doc = build_from_teacher(item, COURSE_META)

        assert isinstance(doc, Document)
        assert doc.metadata["content_type"] == "teacher"
        assert doc.metadata["title"] == "Prof. Smith"
        assert doc.metadata["user_id"] == "teacher_001"
        assert "Teacher: Prof. Smith" in doc.page_content
        assert "Email: smith@university.edu" in doc.page_content


# ---------------------------------------------------------------------------
# Tests — build_from_link_attachment
# ---------------------------------------------------------------------------


class TestBuildFromLinkAttachment:
    """Tests for build_from_link_attachment."""

    def test_youtube_attachment(self) -> None:
        parent = {"id": "cw_001", "title": "Lecture 1"}
        doc = build_from_link_attachment(
            url="https://youtube.com/watch?v=abc123",
            title="Intro to ML",
            attachment_type="youtube",
            extra={"video_id": "abc123"},
            parent_item=parent,
            course_meta=COURSE_META,
            content_type="assignment",
        )

        assert isinstance(doc, Document)
        assert doc.metadata["content_type"] == "assignment_youtube"
        assert doc.metadata["title"] == "Intro to ML"
        assert doc.metadata["url"] == "https://youtube.com/watch?v=abc123"
        assert doc.metadata["attachment_type"] == "youtube"
        assert doc.metadata["video_id"] == "abc123"
        assert doc.metadata["parent_title"] == "Lecture 1"
        assert "YouTube Video: Intro to ML" in doc.page_content
        assert "Video ID: abc123" in doc.page_content

    def test_link_attachment(self) -> None:
        parent = {"id": "ann_001", "text": "Read this article"}
        doc = build_from_link_attachment(
            url="https://example.com/article",
            title="ML Tutorial",
            attachment_type="link",
            extra={},
            parent_item=parent,
            course_meta=COURSE_META,
            content_type="announcement",
        )

        assert doc.metadata["content_type"] == "announcement_link"
        assert doc.metadata["title"] == "ML Tutorial"
        assert doc.metadata["url"] == "https://example.com/article"
        assert doc.metadata["parent_title"] == "Read this article"
        assert "Link: ML Tutorial" in doc.page_content
        assert "URL: https://example.com/article" in doc.page_content

    def test_youtube_with_thumbnail(self) -> None:
        parent = {"id": "cw_002", "title": "Assignment 2"}
        doc = build_from_link_attachment(
            url="https://youtube.com/watch?v=xyz",
            title="Demo Video",
            attachment_type="youtube",
            extra={
                "video_id": "xyz",
                "thumbnail_url": "https://img.youtube.com/thumb.jpg",
            },
            parent_item=parent,
            course_meta=COURSE_META,
            content_type="material",
        )

        assert doc.metadata["thumbnail_url"] == "https://img.youtube.com/thumb.jpg"
        assert doc.metadata["content_type"] == "material_youtube"
