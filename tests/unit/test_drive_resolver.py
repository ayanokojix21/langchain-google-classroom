"""Unit tests for DriveAttachmentResolver."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from langchain_google_classroom.drive_resolver import (
    GOOGLE_EXPORT_MIME_MAP,
    DriveAttachmentResolver,
)

# ---------------------------------------------------------------------------
# Fixtures — sample Classroom item with attachments
# ---------------------------------------------------------------------------

ITEM_WITH_DRIVE_FILES: Dict[str, Any] = {
    "id": "cw_001",
    "title": "Homework 1",
    "materials": [
        {
            "driveFile": {
                "driveFile": {
                    "id": "file_aaa",
                    "title": "Lecture Notes.pdf",
                    "alternateLink": "https://drive.google.com/file/d/file_aaa/view",
                },
                "shareMode": "VIEW",
            }
        },
        {
            "driveFile": {
                "driveFile": {
                    "id": "file_bbb",
                    "title": "Slides.pptx",
                    "alternateLink": "https://drive.google.com/file/d/file_bbb/view",
                },
            }
        },
        {
            "youtubeVideo": {
                "id": "yt_123",
                "title": "Intro Video",
                "alternateLink": "https://youtube.com/watch?v=yt_123",
            }
        },
        {
            "link": {
                "url": "https://example.com",
                "title": "Reference",
            }
        },
    ],
}

ITEM_NO_MATERIALS: Dict[str, Any] = {
    "id": "cw_002",
    "title": "Quiz",
}

ITEM_EMPTY_MATERIALS: Dict[str, Any] = {
    "id": "cw_003",
    "title": "Lab",
    "materials": [],
}


# ---------------------------------------------------------------------------
# Helper to build a resolver with mocked Drive service
# ---------------------------------------------------------------------------


@pytest.fixture
def resolver_with_service() -> tuple[DriveAttachmentResolver, MagicMock]:
    """Create a resolver with a mocked Google Drive service."""
    with (
        patch(
            "langchain_google_classroom.drive_resolver._import_googleapiclient_build"
        ) as mock_build_fn,
        patch(
            "langchain_google_classroom.drive_resolver._import_media_io_base_download"
        ) as mock_download_cls,
    ):
        mock_service = MagicMock()
        mock_build_fn.return_value = MagicMock(return_value=mock_service)

        # Mock MediaIoBaseDownload to write content and return done
        mock_downloader = MagicMock()
        mock_downloader.next_chunk.return_value = (None, True)
        mock_download_cls.return_value = MagicMock(return_value=mock_downloader)

        resolver = DriveAttachmentResolver(credentials=MagicMock())

    return resolver, mock_service


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractDriveFileIds:
    """Tests for extract_drive_file_ids."""

    def test_extracts_drive_files_only(self) -> None:
        """Only driveFile entries should be extracted, not YouTube/links."""
        result = DriveAttachmentResolver.extract_drive_file_ids(ITEM_WITH_DRIVE_FILES)
        assert len(result) == 2
        assert result[0]["id"] == "file_aaa"
        assert result[1]["id"] == "file_bbb"

    def test_no_materials_key(self) -> None:
        """Items without a materials key should return empty list."""
        result = DriveAttachmentResolver.extract_drive_file_ids(ITEM_NO_MATERIALS)
        assert result == []

    def test_empty_materials(self) -> None:
        """Items with empty materials list should return empty list."""
        result = DriveAttachmentResolver.extract_drive_file_ids(ITEM_EMPTY_MATERIALS)
        assert result == []


class TestDriveAttachmentResolver:
    """Tests for the resolve method."""

    def test_resolve_binary_file(
        self,
        resolver_with_service: tuple[DriveAttachmentResolver, MagicMock],
    ) -> None:
        """Non-Google-native file should be downloaded via get_media."""
        resolver, mock_service = resolver_with_service

        # Mock file metadata
        mock_service.files().get().execute.return_value = {
            "id": "file_aaa",
            "name": "Lecture Notes.pdf",
            "mimeType": "application/pdf",
            "webViewLink": "https://drive.google.com/file/d/file_aaa/view",
        }

        item = {
            "id": "cw_001",
            "materials": [
                {
                    "driveFile": {
                        "driveFile": {
                            "id": "file_aaa",
                            "title": "Lecture Notes.pdf",
                        },
                    }
                }
            ],
        }

        attachments = list(resolver.resolve(item))
        assert len(attachments) == 1
        assert attachments[0].file_id == "file_aaa"
        assert attachments[0].title == "Lecture Notes.pdf"
        assert attachments[0].original_mime_type == "application/pdf"

    def test_resolve_google_doc(
        self,
        resolver_with_service: tuple[DriveAttachmentResolver, MagicMock],
    ) -> None:
        """Google Docs should be exported, not downloaded."""
        resolver, mock_service = resolver_with_service

        mock_service.files().get().execute.return_value = {
            "id": "file_doc",
            "name": "My Document",
            "mimeType": "application/vnd.google-apps.document",
            "webViewLink": "https://docs.google.com/document/d/file_doc/edit",
        }

        item = {
            "id": "cw_002",
            "materials": [
                {
                    "driveFile": {
                        "driveFile": {
                            "id": "file_doc",
                            "title": "My Document",
                        },
                    }
                }
            ],
        }

        attachments = list(resolver.resolve(item))
        assert len(attachments) == 1
        assert (
            attachments[0].mime_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert (
            attachments[0].original_mime_type == "application/vnd.google-apps.document"
        )

    def test_resolve_error_skipped(
        self,
        resolver_with_service: tuple[DriveAttachmentResolver, MagicMock],
    ) -> None:
        """Errors during resolution should be logged, not raised."""
        resolver, mock_service = resolver_with_service
        mock_service.files().get().execute.side_effect = Exception("403 Forbidden")

        item = {
            "id": "cw_003",
            "materials": [
                {
                    "driveFile": {
                        "driveFile": {
                            "id": "file_bad",
                            "title": "Secret.pdf",
                        },
                    }
                }
            ],
        }

        # Should not raise
        attachments = list(resolver.resolve(item))
        assert attachments == []

    def test_resolve_no_attachments(
        self,
        resolver_with_service: tuple[DriveAttachmentResolver, MagicMock],
    ) -> None:
        """Item without materials should yield nothing."""
        resolver, _ = resolver_with_service
        attachments = list(resolver.resolve(ITEM_NO_MATERIALS))
        assert attachments == []


class TestGoogleExportMimeMap:
    """Tests for the GOOGLE_EXPORT_MIME_MAP constant."""

    def test_google_docs_exports_text(self) -> None:
        assert (
            GOOGLE_EXPORT_MIME_MAP["application/vnd.google-apps.document"]
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    def test_google_slides_exports_text(self) -> None:
        assert (
            GOOGLE_EXPORT_MIME_MAP["application/vnd.google-apps.presentation"]
            == "application/pdf"
        )

    def test_google_sheets_exports_csv(self) -> None:
        assert (
            GOOGLE_EXPORT_MIME_MAP["application/vnd.google-apps.spreadsheet"]
            == "text/csv"
        )
