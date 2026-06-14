"""Google Classroom API fetcher with paginated data retrieval."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional

from langchain_google_classroom._utilities import (
    _import_googleapiclient_build,
    execute_with_retry,
)

logger = logging.getLogger(__name__)

# Type alias for raw Classroom API response objects
ClassroomObject = Dict[str, Any]


class ClassroomAPIFetcher:
    """Fetches courses, courseWork, announcements, and materials from the
    Google Classroom API.

    All ``list_*`` methods are generators that handle pagination
    transparently via ``nextPageToken``.
    """

    def __init__(self, credentials: Any) -> None:
        """Initialise with Google credentials.

        Args:
            credentials: A ``google.oauth2.credentials.Credentials`` instance.
        """
        build = _import_googleapiclient_build()
        self._service = build("classroom", "v1", credentials=credentials)

    # ------------------------------------------------------------------
    # Courses
    # ------------------------------------------------------------------

    def list_courses(
        self,
        course_ids: Optional[List[str]] = None,
    ) -> Iterator[ClassroomObject]:
        """Yield course dicts.

        If *course_ids* is provided each course is fetched individually via
        ``courses.get``; otherwise all accessible courses are paginated via
        ``courses.list``.

        Args:
            course_ids: Optional list of course IDs to fetch directly.

        Yields:
            Raw course dict from the Classroom API.
        """
        courses = self._service.courses()
        if course_ids:
            for course_id in course_ids:
                try:
                    request = courses.get(id=course_id)
                    course = execute_with_retry(request)
                    yield course
                except Exception as exc:
                    logger.warning("Failed to fetch course %s: %s", course_id, exc)
        else:
            request = courses.list(pageSize=100)
            while request is not None:
                response = execute_with_retry(request)
                for course in response.get("courses", []):
                    yield course
                request = courses.list_next(request, response)

    # ------------------------------------------------------------------
    # CourseWork (assignments)
    # ------------------------------------------------------------------

    def list_course_work(self, course_id: str) -> Iterator[ClassroomObject]:
        """Yield courseWork dicts for *course_id*.

        Args:
            course_id: The course whose work items to fetch.

        Yields:
            Raw courseWork dict.
        """
        try:
            course_work = self._service.courses().courseWork()
            request = course_work.list(courseId=course_id, pageSize=100)
            while request is not None:
                response = execute_with_retry(request)
                for item in response.get("courseWork", []):
                    yield item
                request = course_work.list_next(request, response)
        except Exception as exc:
            logger.warning(
                "Failed to fetch courseWork for course %s: %s",
                course_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Announcements
    # ------------------------------------------------------------------

    def list_announcements(self, course_id: str) -> Iterator[ClassroomObject]:
        """Yield announcement dicts for *course_id*.

        Args:
            course_id: The course whose announcements to fetch.

        Yields:
            Raw announcement dict.
        """
        try:
            announcements = self._service.courses().announcements()
            request = announcements.list(courseId=course_id, pageSize=100)
            while request is not None:
                response = execute_with_retry(request)
                for item in response.get("announcements", []):
                    yield item
                request = announcements.list_next(request, response)
        except Exception as exc:
            logger.warning(
                "Failed to fetch announcements for course %s: %s",
                course_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Course Work Materials
    # ------------------------------------------------------------------

    def list_course_work_materials(self, course_id: str) -> Iterator[ClassroomObject]:
        """Yield courseWorkMaterial dicts for *course_id*.

        Args:
            course_id: The course whose materials to fetch.

        Yields:
            Raw courseWorkMaterial dict.
        """
        try:
            materials = self._service.courses().courseWorkMaterials()
            request = materials.list(courseId=course_id, pageSize=100)
            while request is not None:
                response = execute_with_retry(request)
                for item in response.get("courseWorkMaterial", []):
                    yield item
                request = materials.list_next(request, response)
        except Exception as exc:
            logger.warning(
                "Failed to fetch courseWorkMaterials for course %s: %s",
                course_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Student Submissions
    # ------------------------------------------------------------------

    def list_student_submissions(
        self,
        course_id: str,
        course_work_id: str = "-",
    ) -> Iterator[ClassroomObject]:
        """Yield studentSubmission dicts for *course_id*.

        Args:
            course_id: The course whose submissions to fetch.
            course_work_id: The courseWork to scope submissions to.
                Use ``"-"`` (default) to fetch submissions for **all**
                courseWork in the course.

        Yields:
            Raw studentSubmission dict.
        """
        try:
            submissions = self._service.courses().courseWork().studentSubmissions()
            request = submissions.list(
                courseId=course_id,
                courseWorkId=course_work_id,
                pageSize=100,
            )
            while request is not None:
                response = execute_with_retry(request)
                for item in response.get("studentSubmissions", []):
                    yield item
                request = submissions.list_next(request, response)
        except Exception as exc:
            logger.warning(
                "Failed to fetch studentSubmissions for course %s: %s",
                course_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Rubrics
    # ------------------------------------------------------------------

    def list_rubrics(
        self,
        course_id: str,
        course_work_id: str,
    ) -> Iterator[ClassroomObject]:
        """Yield rubric dicts for a specific courseWork item.

        Rubrics are only available on Google Workspace Education Plus.

        Args:
            course_id: The course containing the courseWork.
            course_work_id: The courseWork whose rubrics to fetch.

        Yields:
            Raw rubric dict.
        """
        try:
            rubrics = self._service.courses().courseWork().rubrics()
            request = rubrics.list(
                courseId=course_id,
                courseWorkId=course_work_id,
                pageSize=100,
            )
            while request is not None:
                response = execute_with_retry(request)
                for item in response.get("rubrics", []):
                    yield item
                request = rubrics.list_next(request, response)
        except Exception as exc:
            logger.warning(
                "Failed to fetch rubrics for course %s / courseWork %s: %s",
                course_id,
                course_work_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Topics
    # ------------------------------------------------------------------

    def list_topics(self, course_id: str) -> Iterator[ClassroomObject]:
        """Yield topic dicts for *course_id*.

        Args:
            course_id: The course whose topics to fetch.

        Yields:
            Raw topic dict.
        """
        try:
            topics = self._service.courses().topics()
            request = topics.list(courseId=course_id, pageSize=100)
            while request is not None:
                response = execute_with_retry(request)
                for item in response.get("topic", []):
                    yield item
                request = topics.list_next(request, response)
        except Exception as exc:
            logger.warning(
                "Failed to fetch topics for course %s: %s",
                course_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Roster — Students
    # ------------------------------------------------------------------

    def list_students(self, course_id: str) -> Iterator[ClassroomObject]:
        """Yield student dicts for *course_id*.

        Each student dict includes a ``profile`` object with
        ``name``, ``emailAddress``, etc.

        Args:
            course_id: The course whose students to fetch.

        Yields:
            Raw student dict.
        """
        try:
            students = self._service.courses().students()
            request = students.list(courseId=course_id, pageSize=100)
            while request is not None:
                response = execute_with_retry(request)
                for item in response.get("students", []):
                    yield item
                request = students.list_next(request, response)
        except Exception as exc:
            logger.warning(
                "Failed to fetch students for course %s: %s",
                course_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Roster — Teachers
    # ------------------------------------------------------------------

    def list_teachers(self, course_id: str) -> Iterator[ClassroomObject]:
        """Yield teacher dicts for *course_id*.

        Each teacher dict includes a ``profile`` object with
        ``name``, ``emailAddress``, etc.

        Args:
            course_id: The course whose teachers to fetch.

        Yields:
            Raw teacher dict.
        """
        try:
            teachers = self._service.courses().teachers()
            request = teachers.list(courseId=course_id, pageSize=100)
            while request is not None:
                response = execute_with_retry(request)
                for item in response.get("teachers", []):
                    yield item
                request = teachers.list_next(request, response)
        except Exception as exc:
            logger.warning(
                "Failed to fetch teachers for course %s: %s",
                course_id,
                exc,
            )
