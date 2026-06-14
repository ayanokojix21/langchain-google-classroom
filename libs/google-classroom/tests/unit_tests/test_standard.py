"""Standard and Pydantic-specific tests for GoogleClassroomLoader.

Verifies the Pydantic v2 model behavior, opt-in scope logic,
async support, and serialization — all features introduced in v0.2.0.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.document_loaders import BaseLoader
from pydantic import BaseModel

from langchain_google_classroom._utilities import DEFAULT_SCOPES
from langchain_google_classroom.loader import (
    ROSTERS_SCOPE,
    SUBMISSIONS_SCOPE,
    TOPICS_SCOPE,
    GoogleClassroomLoader,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_loader(**kwargs: Any) -> GoogleClassroomLoader:
    """Create a loader with a dummy credentials object."""
    return GoogleClassroomLoader(credentials=MagicMock(), **kwargs)


# ---------------------------------------------------------------------------
# Pydantic v2 model tests
# ---------------------------------------------------------------------------


class TestPydanticModel:
    """Verify the loader is a proper Pydantic v2 BaseModel."""

    def test_is_pydantic_base_model(self) -> None:
        """GoogleClassroomLoader should be a Pydantic BaseModel."""
        loader = _make_loader()
        assert isinstance(loader, BaseModel)

    def test_is_langchain_base_loader(self) -> None:
        """GoogleClassroomLoader should also be a LangChain BaseLoader."""
        loader = _make_loader()
        assert isinstance(loader, BaseLoader)

    def test_default_field_values(self) -> None:
        """Default field values should match expected defaults."""
        loader = _make_loader()
        assert loader.load_assignments is True
        assert loader.load_announcements is True
        assert loader.load_materials is True
        assert loader.load_attachments is True
        assert loader.parse_attachments is True
        assert loader.load_images is False
        assert loader.load_submissions is False
        assert loader.load_topics is False
        assert loader.load_roster is False
        assert loader.max_file_size == 50_000_000
        assert loader.vision_model is None
        assert loader.image_prompt is None
        assert loader.file_parser_cls is None
        assert loader.file_parser_kwargs == {}
        assert loader.course_ids is None

    def test_field_override(self) -> None:
        """Fields should be overridable via keyword args."""
        loader = _make_loader(
            course_ids=["abc"],
            load_assignments=False,
            load_submissions=True,
            max_file_size=10_000,
        )
        assert loader.course_ids == ["abc"]
        assert loader.load_assignments is False
        assert loader.load_submissions is True
        assert loader.max_file_size == 10_000

    def test_model_json_schema(self) -> None:
        """The model should produce a valid JSON schema."""
        schema = GoogleClassroomLoader.model_json_schema()
        assert "properties" in schema
        assert "course_ids" in schema["properties"]
        assert "load_assignments" in schema["properties"]
        assert "load_submissions" in schema["properties"]

    def test_credentials_not_serialized(self) -> None:
        """Credentials should be excluded from serialized output."""
        loader = _make_loader()
        data = loader.model_dump()
        assert "load_assignments" in data
        # credentials must be excluded to prevent leaking secrets
        assert "credentials" not in data


# ---------------------------------------------------------------------------
# Opt-in scopes tests
# ---------------------------------------------------------------------------


class TestOptInScopes:
    """Verify that scopes are dynamically built from feature flags."""

    def test_default_scopes_unchanged(self) -> None:
        """With no opt-in features, scopes should equal DEFAULT_SCOPES."""
        loader = _make_loader()
        assert loader.scopes == DEFAULT_SCOPES

    def test_submissions_scope_added(self) -> None:
        """Enabling load_submissions should add the submissions scope."""
        loader = _make_loader(load_submissions=True)
        assert SUBMISSIONS_SCOPE in loader.scopes
        # Default scopes still present
        for scope in DEFAULT_SCOPES:
            assert scope in loader.scopes

    def test_roster_scope_added(self) -> None:
        """Enabling load_roster should add the roster scope."""
        loader = _make_loader(load_roster=True)
        assert ROSTERS_SCOPE in loader.scopes

    def test_topics_scope_added(self) -> None:
        """Enabling load_topics should add the topics scope."""
        loader = _make_loader(load_topics=True)
        assert TOPICS_SCOPE in loader.scopes

    def test_all_optional_scopes_added(self) -> None:
        """Enabling all features should add all optional scopes."""
        loader = _make_loader(
            load_submissions=True,
            load_topics=True,
            load_roster=True,
        )
        assert SUBMISSIONS_SCOPE in loader.scopes
        assert ROSTERS_SCOPE in loader.scopes
        assert TOPICS_SCOPE in loader.scopes

    def test_no_duplicate_scopes(self) -> None:
        """Scopes should not contain duplicates."""
        loader = _make_loader(load_submissions=True)
        assert len(loader.scopes) == len(set(loader.scopes))

    def test_custom_scopes_preserved(self) -> None:
        """User-provided scopes should not be overwritten."""
        custom = ["https://example.com/custom.scope"]
        loader = _make_loader(scopes=custom, load_submissions=True)
        assert "https://example.com/custom.scope" in loader.scopes
        assert SUBMISSIONS_SCOPE in loader.scopes

    def test_disabled_features_no_extra_scopes(self) -> None:
        """Disabled features should NOT add their scopes."""
        loader = _make_loader(
            load_submissions=False,
            load_topics=False,
            load_roster=False,
        )
        assert SUBMISSIONS_SCOPE not in loader.scopes
        assert ROSTERS_SCOPE not in loader.scopes
        assert TOPICS_SCOPE not in loader.scopes


# ---------------------------------------------------------------------------
# Async support tests
# ---------------------------------------------------------------------------


@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
class TestAsyncSupport:
    """Verify alazy_load() works correctly.

    Note: asyncio event loop on Windows requires socket access for its
    internal self-pipe, so we allow localhost connections.
    """

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_alazy_load_returns_async_iterator(
        self, mock_fetcher_cls: MagicMock
    ) -> None:
        """alazy_load should return an async iterator."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([{"id": "1", "name": "Test"}])
        fetcher.list_course_work.return_value = iter(
            [
                {
                    "id": "cw1",
                    "title": "HW1",
                    "description": "Desc",
                    "state": "PUBLISHED",
                    "creationTime": "2024-01-01T00:00:00Z",
                    "updateTime": "2024-01-01T00:00:00Z",
                    "alternateLink": "https://example.com",
                }
            ]
        )
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])

        loader = _make_loader(
            course_ids=["1"],
            load_attachments=False,
        )

        async def _run() -> list:
            return [doc async for doc in loader.alazy_load()]

        docs = asyncio.run(_run())
        assert len(docs) == 1
        assert docs[0].metadata["content_type"] == "assignment"

    @patch(
        "langchain_google_classroom.loader.ClassroomAPIFetcher",
        autospec=True,
    )
    def test_aload_returns_list(self, mock_fetcher_cls: MagicMock) -> None:
        """aload() should return a list of Documents."""
        fetcher = mock_fetcher_cls.return_value
        fetcher.list_courses.return_value = iter([{"id": "1", "name": "Test"}])
        fetcher.list_course_work.return_value = iter([])
        fetcher.list_announcements.return_value = iter([])
        fetcher.list_course_work_materials.return_value = iter([])

        loader = _make_loader(
            course_ids=["1"],
            load_attachments=False,
        )

        docs = asyncio.run(loader.aload())
        assert isinstance(docs, list)
        assert docs == []


# ---------------------------------------------------------------------------
# Standard loader interface tests
# ---------------------------------------------------------------------------


class TestStandardInterface:
    """Verify the loader conforms to LangChain's expected interface."""

    def test_has_lazy_load(self) -> None:
        """The loader must have a lazy_load method."""
        assert hasattr(GoogleClassroomLoader, "lazy_load")

    def test_has_alazy_load(self) -> None:
        """The loader must have an alazy_load method."""
        assert hasattr(GoogleClassroomLoader, "alazy_load")

    def test_has_load(self) -> None:
        """The loader must have a load method (from BaseLoader)."""
        assert hasattr(GoogleClassroomLoader, "load")

    def test_has_aload(self) -> None:
        """The loader must have an aload method (from BaseLoader)."""
        assert hasattr(GoogleClassroomLoader, "aload")

    def test_load_calls_lazy_load(self) -> None:
        """load() should delegate to lazy_load() under the hood."""
        with patch.object(
            GoogleClassroomLoader,
            "lazy_load",
            return_value=iter([]),
        ) as mock_lazy:
            loader = _make_loader()
            result = loader.load()
            assert result == []
            mock_lazy.assert_called_once()
