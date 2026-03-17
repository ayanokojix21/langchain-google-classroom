"""Resolve and download Google Drive attachments from Classroom items."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Dict, Iterator, List, NamedTuple

from langchain_google_classroom._utilities import (
    _import_googleapiclient_build,
    _import_media_io_base_download,
    execute_with_retry,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google-native MIME types → export targets
# ---------------------------------------------------------------------------

GOOGLE_EXPORT_MIME_MAP: Dict[str, str] = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    "application/vnd.google-apps.presentation": "application/pdf",
    "application/vnd.google-apps.spreadsheet": "text/csv",
}

# ---------------------------------------------------------------------------
# Data container for a resolved attachment
# ---------------------------------------------------------------------------


class ResolvedAttachment(NamedTuple):
    """A Drive attachment whose content has been downloaded or exported.

    Attributes:
        file_id: Google Drive file ID.
        title: Human-readable file name.
        mime_type: MIME type of the *downloaded* content (may differ from
            the original for Google-native files that were exported).
        content: Raw file bytes.
        source_url: webViewLink or alternateLink for the file.
        original_mime_type: The file's original MIME type on Drive.
    """

    file_id: str
    title: str
    mime_type: str
    content: bytes
    source_url: str
    original_mime_type: str


# ---------------------------------------------------------------------------
# Resolver class
# ---------------------------------------------------------------------------


class DriveAttachmentResolver:
    """Download or export Google Drive files attached to Classroom items.

    For **Google-native** files (Docs, Slides, Sheets), the resolver uses
    the Drive ``files.export`` endpoint to obtain parser-friendly exports:
    DOCX for Google Docs, PDF for Google Slides, and CSV for Sheets.

    For **binary** files (PDF, Docx, images, etc.), the resolver uses the
    Drive ``files.get_media`` endpoint to stream the raw bytes.
    """

    def __init__(self, credentials: Any) -> None:
        """Initialise the resolver with Google credentials.

        Args:
            credentials: A ``google.oauth2`` credentials object.
        """
        build = _import_googleapiclient_build()
        self._service = build("drive", "v3", credentials=credentials)
        self._MediaIoBaseDownload = _import_media_io_base_download()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def extract_drive_file_ids(
        item: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract Drive file entries from a Classroom item.

        Classroom items store attachments under a ``materials`` key.  Each
        material entry may contain a ``driveFile``, ``youtubeVideo``,
        ``link``, or ``form``.  Only ``driveFile`` entries are extracted.

        Args:
            item: A raw courseWork, announcement, or courseWorkMaterial dict.

        Returns:
            List of dicts, each with ``id``, ``title``, and
            ``alternateLink`` keys for a Drive file.
        """
        drive_files: List[Dict[str, Any]] = []
        for material in item.get("materials", []):
            drive_entry = material.get("driveFile")
            if drive_entry:
                inner = drive_entry.get("driveFile", {})
                if inner.get("id"):
                    drive_files.append(inner)
        return drive_files

    def resolve(self, item: Dict[str, Any]) -> Iterator[ResolvedAttachment]:
        """Resolve all Drive attachments in a Classroom item.

        For each Drive file found in ``item["materials"]``:

        1. Fetch file metadata (name, mimeType, webViewLink).
          2. If the file is a Google-native type (Docs/Slides/Sheets),
              **export** it based on ``GOOGLE_EXPORT_MIME_MAP``.
        3. Otherwise, **download** the raw binary content.

        Errors for individual files are logged and skipped.

        Args:
            item: A raw courseWork, announcement, or courseWorkMaterial dict.

        Yields:
            :class:`ResolvedAttachment` for each successfully resolved file.
        """
        drive_files = self.extract_drive_file_ids(item)
        for drive_file in drive_files:
            file_id = drive_file.get("id", "")
            try:
                metadata = self._get_file_metadata(file_id)
                original_mime = metadata.get("mimeType", "")
                title = metadata.get("name", drive_file.get("title", ""))
                source_url = metadata.get(
                    "webViewLink",
                    drive_file.get("alternateLink", ""),
                )

                export_mime = GOOGLE_EXPORT_MIME_MAP.get(original_mime)
                if export_mime:
                    content = self._export_google_doc(file_id, export_mime)
                    resolved_mime = export_mime
                else:
                    content = self._download_file(file_id)
                    resolved_mime = original_mime

                yield ResolvedAttachment(
                    file_id=file_id,
                    title=title,
                    mime_type=resolved_mime,
                    content=content,
                    source_url=source_url,
                    original_mime_type=original_mime,
                )
            except Exception as exc:
                logger.warning("Failed to resolve Drive file %s: %s", file_id, exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """Fetch metadata for a Drive file.

        Args:
            file_id: The Drive file ID.

        Returns:
            Dict with ``id``, ``name``, ``mimeType``, and ``webViewLink``.
        """
        return execute_with_retry(
            self._service.files().get(
                fileId=file_id,
                supportsAllDrives=True,
                fields="id,name,mimeType,webViewLink",
            )
        )

    def _export_google_doc(self, file_id: str, export_mime: str) -> bytes:
        """Export a Google-native file (Docs/Slides/Sheets) to *export_mime*.

        Uses the Drive ``files.export_media`` endpoint with chunked
        streaming via ``MediaIoBaseDownload``.

        Args:
            file_id: The Drive file ID.
            export_mime: Target MIME type (e.g. ``"application/pdf"``).

        Returns:
            Exported content as bytes.
        """
        request = self._service.files().export_media(
            fileId=file_id, mimeType=export_mime
        )
        return self._stream_download(request)

    def _download_file(self, file_id: str) -> bytes:
        """Download a binary file from Drive.

        Uses the Drive ``files.get_media`` endpoint with chunked
        streaming via ``MediaIoBaseDownload``.

        Args:
            file_id: The Drive file ID.

        Returns:
            Raw file bytes.
        """
        request = self._service.files().get_media(fileId=file_id)
        return self._stream_download(request)

    def _stream_download(self, request: Any) -> bytes:
        """Stream a download request into memory.

        Args:
            request: A ``googleapiclient`` media request.

        Returns:
            Downloaded bytes.
        """
        fh = BytesIO()
        downloader = self._MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return fh.getvalue()
