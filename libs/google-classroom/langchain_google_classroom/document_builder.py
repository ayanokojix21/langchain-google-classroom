"""Build LangChain Document objects from raw Classroom API responses."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.documents import Document

from langchain_google_classroom.normalizer import normalize

logger = logging.getLogger(__name__)

# Type alias for raw Classroom API response objects
ClassroomObject = Dict[str, Any]

# Source identifier for all LangChain Document metadata
SOURCE = "google_classroom"

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def build_course_meta(course: ClassroomObject) -> Dict[str, Any]:
    """Extract reusable metadata from a course dict.

    Args:
        course: Raw course dict from the Classroom API.

    Returns:
        Dict with ``course_id`` and ``course_name``.
    """
    return {
        "course_id": course.get("id", ""),
        "course_name": course.get("name", ""),
    }


def _format_due_date(item: ClassroomObject) -> Optional[str]:
    """Build an ISO-style due-date string from *dueDate* and *dueTime* fields.

    Args:
        item: A courseWork dict that may contain ``dueDate`` and ``dueTime``.

    Returns:
        A string like ``"2024-01-22T23:59:00"`` or ``None`` if no due date is
        set.
    """
    due_date = item.get("dueDate")
    if not due_date:
        return None

    year = due_date.get("year", 0)
    month = due_date.get("month", 1)
    day = due_date.get("day", 1)
    date_str = f"{year:04d}-{month:02d}-{day:02d}"

    due_time = item.get("dueTime")
    if due_time:
        hours = due_time.get("hours", 0)
        minutes = due_time.get("minutes", 0)
        date_str += f"T{hours:02d}:{minutes:02d}:00"

    return date_str


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------


def build_from_course_work(
    item: ClassroomObject,
    course_meta: Dict[str, Any],
) -> Document:
    """Convert a courseWork dict into a :class:`Document`.

    Args:
        item: Raw courseWork dict from the Classroom API.
        course_meta: Dict with ``course_id`` and ``course_name``.

    Returns:
        A LangChain ``Document`` with assignment content and metadata.
    """
    title = item.get("title", "Untitled Assignment")
    description = item.get("description", "")

    # Build page content
    parts = [f"Assignment: {title}"]
    if description:
        parts.append("")
        parts.append(description)
    page_content = normalize("\n".join(parts))

    metadata: Dict[str, Any] = {
        "source": SOURCE,
        **course_meta,
        "content_type": "assignment",
        "title": title,
        "item_id": item.get("id", ""),
        "created_time": item.get("creationTime", ""),
        "updated_time": item.get("updateTime", ""),
        "alternate_link": item.get("alternateLink", ""),
        "state": item.get("state", ""),
    }

    due_date = _format_due_date(item)
    if due_date:
        metadata["due_date"] = due_date

    max_points = item.get("maxPoints")
    if max_points is not None:
        metadata["max_points"] = float(max_points)

    return Document(page_content=page_content, metadata=metadata)


def build_from_announcement(
    item: ClassroomObject,
    course_meta: Dict[str, Any],
) -> Document:
    """Convert an announcement dict into a :class:`Document`.

    Args:
        item: Raw announcement dict from the Classroom API.
        course_meta: Dict with ``course_id`` and ``course_name``.

    Returns:
        A LangChain ``Document`` with announcement content and metadata.
    """
    text = item.get("text", "")
    # Use first 80 chars as the title, cleaning newlines for safer titles
    title = (
        text.replace("\n", " ").replace("\r", "")[:80].strip()
        if text
        else "Untitled Announcement"
    )

    page_content = normalize(f"Announcement: {text}")

    metadata: Dict[str, Any] = {
        "source": SOURCE,
        **course_meta,
        "content_type": "announcement",
        "title": title,
        "item_id": item.get("id", ""),
        "created_time": item.get("creationTime", ""),
        "updated_time": item.get("updateTime", ""),
        "alternate_link": item.get("alternateLink", ""),
        "state": item.get("state", ""),
    }

    return Document(page_content=page_content, metadata=metadata)


def build_from_material(
    item: ClassroomObject,
    course_meta: Dict[str, Any],
) -> Document:
    """Convert a courseWorkMaterial dict into a :class:`Document`.

    Args:
        item: Raw courseWorkMaterial dict from the Classroom API.
        course_meta: Dict with ``course_id`` and ``course_name``.

    Returns:
        A LangChain ``Document`` with material content and metadata.
    """
    title = item.get("title", "Untitled Material")
    description = item.get("description", "")

    parts = [f"Material: {title}"]
    if description:
        parts.append("")
        parts.append(description)
    page_content = normalize("\n".join(parts))

    metadata: Dict[str, Any] = {
        "source": SOURCE,
        **course_meta,
        "content_type": "material",
        "title": title,
        "item_id": item.get("id", ""),
        "created_time": item.get("creationTime", ""),
        "updated_time": item.get("updateTime", ""),
        "alternate_link": item.get("alternateLink", ""),
        "state": item.get("state", ""),
    }

    return Document(page_content=page_content, metadata=metadata)


def build_from_attachment(
    file_id: str,
    title: str,
    mime_type: str,
    source_url: str,
    original_mime_type: str,
    parsed_text: str,
    parent_item: ClassroomObject,
    course_meta: Dict[str, Any],
    content_type: str,
) -> Document:
    """Build a :class:`Document` from a resolved and parsed Drive attachment.

    Args:
        file_id: Google Drive file ID.
        title: File name.
        mime_type: MIME type of the downloaded/exported content.
        source_url: Web link to the file.
        original_mime_type: Original MIME type on Drive.
        parsed_text: Text extracted from the file by a parser.
        parent_item: The parent courseWork / announcement / material dict.
        course_meta: Dict with ``course_id`` and ``course_name``.
        content_type: Parent content type (``"assignment"``, ``"announcement"``,
            or ``"material"``).

    Returns:
        A LangChain ``Document`` with attachment content and metadata.
    """
    parent_text = parent_item.get("text", "").replace("\n", " ").replace("\r", "")
    parent_title = parent_item.get("title") or parent_text[:80].strip()

    page_content = normalize(parsed_text)

    metadata: Dict[str, Any] = {
        "source": SOURCE,
        **course_meta,
        "content_type": f"{content_type}_attachment",
        "title": title,
        "item_id": parent_item.get("id", ""),
        "parent_title": parent_title,
        "file_id": file_id,
        "mime_type": original_mime_type,
        "attachment_url": source_url,
        "created_time": parent_item.get("creationTime", ""),
        "updated_time": parent_item.get("updateTime", ""),
        "alternate_link": parent_item.get("alternateLink", ""),
    }

    return Document(page_content=page_content, metadata=metadata)


# ---------------------------------------------------------------------------
# Student Submissions
# ---------------------------------------------------------------------------


def build_from_submission(
    item: ClassroomObject,
    course_meta: Dict[str, Any],
) -> Document:
    """Convert a studentSubmission dict into a :class:`Document`.

    Args:
        item: Raw studentSubmission dict from the Classroom API.
        course_meta: Dict with ``course_id`` and ``course_name``.

    Returns:
        A LangChain ``Document`` with submission content and metadata.
    """
    state = item.get("state", "UNKNOWN")
    course_work_id = item.get("courseWorkId", "")
    user_id = item.get("userId", "")

    # Build page content from submission data
    parts = [f"Student Submission (courseWork: {course_work_id})"]
    parts.append(f"State: {state}")

    if item.get("late"):
        parts.append("Late: Yes")

    # Short answer submission
    short_answer = item.get("shortAnswerSubmission")
    if short_answer:
        parts.append(f"Answer: {short_answer.get('answer', '')}")

    # Multiple choice submission
    mc_answer = item.get("multipleChoiceSubmission")
    if mc_answer:
        parts.append(f"Answer: {mc_answer.get('answer', '')}")

    # Assignment submission (file attachments)
    assignment_sub = item.get("assignmentSubmission")
    if assignment_sub:
        attachments = assignment_sub.get("attachments", [])
        if attachments:
            titles = []
            for att in attachments:
                drive_file = att.get("driveFile", {})
                titles.append(drive_file.get("title", "Untitled"))
            parts.append(f"Attachments: {', '.join(titles)}")

    assigned_grade = item.get("assignedGrade")
    if assigned_grade is not None:
        parts.append(f"Grade: {assigned_grade}")

    page_content = normalize("\n".join(parts))

    metadata: Dict[str, Any] = {
        "source": SOURCE,
        **course_meta,
        "content_type": "submission",
        "item_id": item.get("id", ""),
        "course_work_id": course_work_id,
        "user_id": user_id,
        "state": state,
        "late": item.get("late", False),
        "created_time": item.get("creationTime", ""),
        "updated_time": item.get("updateTime", ""),
        "alternate_link": item.get("alternateLink", ""),
    }

    if assigned_grade is not None:
        metadata["assigned_grade"] = float(assigned_grade)

    draft_grade = item.get("draftGrade")
    if draft_grade is not None:
        metadata["draft_grade"] = float(draft_grade)

    return Document(page_content=page_content, metadata=metadata)


# ---------------------------------------------------------------------------
# Rubrics
# ---------------------------------------------------------------------------


def _format_rubric_criteria(criteria: list) -> str:
    """Format rubric criteria and levels into readable text.

    Args:
        criteria: List of criterion dicts from the rubric.

    Returns:
        Human-readable formatted text.
    """
    parts: list[str] = []
    for i, criterion in enumerate(criteria, 1):
        title = criterion.get("title", f"Criterion {i}")
        description = criterion.get("description", "")
        parts.append(f"  Criterion {i}: {title}")
        if description:
            parts.append(f"    Description: {description}")

        levels = criterion.get("levels", [])
        for level in levels:
            level_title = level.get("title", "")
            level_desc = level.get("description", "")
            points = level.get("points")
            level_text = f"    - {level_title}"
            if points is not None:
                level_text += f" ({points} pts)"
            if level_desc:
                level_text += f": {level_desc}"
            parts.append(level_text)

    return "\n".join(parts)


def build_from_rubric(
    item: ClassroomObject,
    course_meta: Dict[str, Any],
    course_work_title: str = "",
) -> Document:
    """Convert a rubric dict into a :class:`Document`.

    Args:
        item: Raw rubric dict from the Classroom API.
        course_meta: Dict with ``course_id`` and ``course_name``.
        course_work_title: Title of the parent courseWork (for context).

    Returns:
        A LangChain ``Document`` with rubric content and metadata.
    """
    course_work_id = item.get("courseWorkId", "")

    parts = [f"Rubric for: {course_work_title or course_work_id}"]

    criteria = item.get("criteria", [])
    if criteria:
        parts.append(_format_rubric_criteria(criteria))

    page_content = normalize("\n".join(parts))

    metadata: Dict[str, Any] = {
        "source": SOURCE,
        **course_meta,
        "content_type": "rubric",
        "item_id": item.get("id", ""),
        "course_work_id": course_work_id,
        "created_time": item.get("creationTime", ""),
        "updated_time": item.get("updateTime", ""),
        "criteria_count": len(criteria),
    }

    if course_work_title:
        metadata["course_work_title"] = course_work_title

    return Document(page_content=page_content, metadata=metadata)


# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------


def build_from_topic(
    item: ClassroomObject,
    course_meta: Dict[str, Any],
) -> Document:
    """Convert a topic dict into a :class:`Document`.

    Args:
        item: Raw topic dict from the Classroom API.
        course_meta: Dict with ``course_id`` and ``course_name``.

    Returns:
        A LangChain ``Document`` with topic content and metadata.
    """
    name = item.get("name", "Untitled Topic")

    page_content = normalize(f"Topic: {name}")

    metadata: Dict[str, Any] = {
        "source": SOURCE,
        **course_meta,
        "content_type": "topic",
        "title": name,
        "item_id": item.get("topicId", ""),
        "updated_time": item.get("updateTime", ""),
    }

    return Document(page_content=page_content, metadata=metadata)


# ---------------------------------------------------------------------------
# Roster — Students & Teachers
# ---------------------------------------------------------------------------


def _extract_profile_name(profile: Dict[str, Any]) -> str:
    """Extract a full name from a user profile dict.

    Args:
        profile: User profile dict with a ``name`` sub-dict.

    Returns:
        Full name string, or ``"Unknown"`` if not available.
    """
    name = profile.get("name", {})
    full_name = name.get("fullName", "")
    if full_name:
        return full_name
    given = name.get("givenName", "")
    family = name.get("familyName", "")
    return f"{given} {family}".strip() or "Unknown"


def build_from_student(
    item: ClassroomObject,
    course_meta: Dict[str, Any],
) -> Document:
    """Convert a student dict into a :class:`Document`.

    Args:
        item: Raw student dict from the Classroom API.
        course_meta: Dict with ``course_id`` and ``course_name``.

    Returns:
        A LangChain ``Document`` with student profile content and metadata.
    """
    profile = item.get("profile", {})
    name = _extract_profile_name(profile)
    email = profile.get("emailAddress", "")
    user_id = item.get("userId", "")

    parts = [f"Student: {name}"]
    if email:
        parts.append(f"Email: {email}")

    page_content = normalize("\n".join(parts))

    metadata: Dict[str, Any] = {
        "source": SOURCE,
        **course_meta,
        "content_type": "student",
        "title": name,
        "user_id": user_id,
        "email": email,
    }

    return Document(page_content=page_content, metadata=metadata)


def build_from_teacher(
    item: ClassroomObject,
    course_meta: Dict[str, Any],
) -> Document:
    """Convert a teacher dict into a :class:`Document`.

    Args:
        item: Raw teacher dict from the Classroom API.
        course_meta: Dict with ``course_id`` and ``course_name``.

    Returns:
        A LangChain ``Document`` with teacher profile content and metadata.
    """
    profile = item.get("profile", {})
    name = _extract_profile_name(profile)
    email = profile.get("emailAddress", "")
    user_id = item.get("userId", "")

    parts = [f"Teacher: {name}"]
    if email:
        parts.append(f"Email: {email}")

    page_content = normalize("\n".join(parts))

    metadata: Dict[str, Any] = {
        "source": SOURCE,
        **course_meta,
        "content_type": "teacher",
        "title": name,
        "user_id": user_id,
        "email": email,
    }

    return Document(page_content=page_content, metadata=metadata)


# ---------------------------------------------------------------------------
# Link / YouTube attachment builder
# ---------------------------------------------------------------------------


def build_from_link_attachment(
    url: str,
    title: str,
    attachment_type: str,
    extra: Dict[str, str],
    parent_item: ClassroomObject,
    course_meta: Dict[str, Any],
    content_type: str,
) -> Document:
    """Build a Document from a YouTube or external link attachment.

    Unlike Drive files, these are not downloaded — only their metadata
    is captured as structured text.

    Args:
        url: The link or YouTube URL.
        title: Human-readable title.
        attachment_type: ``"youtube"`` or ``"link"``.
        extra: Additional metadata (e.g. ``video_id``, ``thumbnail_url``).
        parent_item: The parent Classroom item dict.
        course_meta: Dict with ``course_id`` and ``course_name``.
        content_type: Parent content type (e.g. ``"assignment"``).

    Returns:
        A LangChain ``Document`` with link metadata.
    """
    parent_title = parent_item.get("title", parent_item.get("text", "")[:80])

    parts = []
    if attachment_type == "youtube":
        parts.append(f"YouTube Video: {title}")
        parts.append(f"URL: {url}")
        video_id = extra.get("video_id", "")
        if video_id:
            parts.append(f"Video ID: {video_id}")
    else:
        parts.append(f"Link: {title}")
        parts.append(f"URL: {url}")

    page_content = normalize("\n".join(parts))

    metadata: Dict[str, Any] = {
        "source": SOURCE,
        **course_meta,
        "content_type": f"{content_type}_{attachment_type}",
        "title": title,
        "url": url,
        "attachment_type": attachment_type,
        "item_id": parent_item.get("id", ""),
        "parent_title": parent_title,
        **extra,
    }

    return Document(page_content=page_content, metadata=metadata)
