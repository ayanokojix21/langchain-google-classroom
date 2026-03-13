"""Build LangChain Document objects from raw Classroom API responses."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Optional

from langchain_core.documents import Document

from langchain_google_classroom.normalizer import normalize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def build_course_meta(course: Dict[str, Any]) -> Dict[str, Any]:
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


def _format_due_date(item: Dict[str, Any]) -> Optional[str]:
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
    item: Dict[str, Any],
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
        "source": "google_classroom",
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
    item: Dict[str, Any],
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
    # Use first 80 chars as the title
    title = text[:80].strip() if text else "Untitled Announcement"

    page_content = normalize(f"Announcement: {text}")

    metadata: Dict[str, Any] = {
        "source": "google_classroom",
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
    item: Dict[str, Any],
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
        "source": "google_classroom",
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