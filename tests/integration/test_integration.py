"""Integration tests for langchain-google-classroom.

These tests exercise the **real** parser and builder logic (no mocks
on the parsers themselves) and include skipped-by-default tests that
hit the live Google Classroom API when credentials are available.

Run unit tests only (default):
    pytest tests/ -m "not integration"

Run integration tests (requires credentials):
    pytest tests/ -m integration
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import TypedDict
from unittest.mock import MagicMock

import pytest
from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents import Document
from langchain_core.documents.base import Blob

from langchain_google_classroom.document_builder import (
    build_course_meta,
    build_from_announcement,
    build_from_attachment,
    build_from_course_work,
    build_from_material,
)
from langchain_google_classroom.loader import GoogleClassroomLoader
from langchain_google_classroom.normalizer import normalize
from langchain_google_classroom.parsers import ImageParser, get_parser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_COURSE = {"id": "111", "name": "Integration Test Course"}


def _make_simple_pdf() -> bytes:
    """Build a minimal valid PDF in memory using pypdf."""
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _make_simple_docx() -> bytes:
    """Build a minimal valid DOCX in memory using python-docx."""
    docx = pytest.importorskip("docx")

    doc = docx.Document()
    doc.add_paragraph("Integration test paragraph one")
    doc.add_paragraph("Integration test paragraph two")
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Integration: real PDF parsing (no mocks on pypdf)
# ---------------------------------------------------------------------------


class TestRealPDFParsing:
    """Test PDFParser with actual pypdf — no mocking."""

    @pytest.mark.integration
    def test_parse_real_pdf(self) -> None:
        """A real blank PDF should parse without errors."""
        pdf_bytes = _make_simple_pdf()
        parser = get_parser("application/pdf")
        assert parser is not None

        blob = Blob(data=pdf_bytes, mimetype="application/pdf")
        docs = list(parser.lazy_parse(blob))
        # Blank page has no text, so 0 docs is valid
        assert isinstance(docs, list)

    @pytest.mark.integration
    def test_pdf_parser_type(self) -> None:
        """Registry should return PDFParser for application/pdf."""
        from langchain_google_classroom.parsers import PDFParser

        parser = get_parser("application/pdf")
        assert isinstance(parser, PDFParser)
        assert isinstance(parser, BaseBlobParser)


# ---------------------------------------------------------------------------
# Integration: real DOCX parsing (no mocks on python-docx)
# ---------------------------------------------------------------------------


class TestRealDOCXParsing:
    """Test DocxParser with actual python-docx — no mocking."""

    @pytest.mark.integration
    def test_parse_real_docx(self) -> None:
        """A real DOCX with paragraphs should yield text."""
        docx_bytes = _make_simple_docx()
        parser = get_parser(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert parser is not None

        blob = Blob(data=docx_bytes, mimetype="application/octet-stream")
        docs = list(parser.lazy_parse(blob))

        assert len(docs) == 1
        assert "Integration test paragraph one" in docs[0].page_content
        assert "Integration test paragraph two" in docs[0].page_content

    @pytest.mark.integration
    def test_docx_parser_type(self) -> None:
        """Registry should return DocxParser for DOCX MIME."""
        from langchain_google_classroom.parsers import DocxParser

        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        parser = get_parser(mime)
        assert isinstance(parser, DocxParser)
        assert isinstance(parser, BaseBlobParser)


# ---------------------------------------------------------------------------
# Integration: text parsing
# ---------------------------------------------------------------------------


class TestRealTextParsing:
    """Test TextParser end-to-end."""

    @pytest.mark.integration
    def test_csv_content(self) -> None:
        """CSV content should be parsed as plain text."""
        csv_data = b"name,score\nAlice,95\nBob,88\n"
        parser = get_parser("text/csv")
        assert parser is not None

        blob = Blob(data=csv_data, mimetype="text/csv")
        docs = list(parser.lazy_parse(blob))

        assert len(docs) == 1
        assert "Alice,95" in docs[0].page_content
        assert "Bob,88" in docs[0].page_content

    @pytest.mark.integration
    def test_html_content(self) -> None:
        """HTML content should be parsed as raw text."""
        html = b"<html><body><h1>Hello</h1><p>World</p></body></html>"
        parser = get_parser("text/html")
        assert parser is not None

        blob = Blob(data=html, mimetype="text/html")
        docs = list(parser.lazy_parse(blob))

        assert len(docs) == 1
        assert "Hello" in docs[0].page_content


# ---------------------------------------------------------------------------
# Integration: image parser with mock vision model
# ---------------------------------------------------------------------------


class TestImageParserIntegration:
    """Test ImageParser with a mocked vision model end-to-end."""

    @pytest.mark.integration
    def test_image_with_mock_vision(self) -> None:
        """ImageParser should invoke the vision model and return
        its description."""
        # Create a fake PNG (magic bytes + garbage)
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "A diagram showing the water cycle with arrows."
        mock_model.invoke.return_value = mock_response

        parser = ImageParser(vision_model=mock_model)
        blob = Blob(data=png_bytes, mimetype="image/png")
        docs = list(parser.lazy_parse(blob))

        assert len(docs) == 1
        assert "water cycle" in docs[0].page_content
        assert "image_base64" not in docs[0].metadata
        mock_model.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# Integration: document builder pipeline
# ---------------------------------------------------------------------------


class TestDocumentBuilderPipeline:
    """Test the full builder → normalizer pipeline."""

    @pytest.mark.integration
    def test_full_assignment_pipeline(self) -> None:
        """Build a Document from raw API data and verify all metadata."""
        course_meta = build_course_meta(SAMPLE_COURSE)
        raw_item = {
            "id": "hw1",
            "title": "Final Project",
            "description": "Build a   \r\n  machine learning\r\n\n\n\nmodel",
            "creationTime": "2024-01-15T10:00:00Z",
            "updateTime": "2024-01-16T12:00:00Z",
            "alternateLink": "https://classroom.google.com/c/111/a/hw1",
            "state": "PUBLISHED",
            "maxPoints": 100,
            "dueDate": {"year": 2024, "month": 3, "day": 15},
            "dueTime": {"hours": 23, "minutes": 59},
        }

        doc = build_from_course_work(raw_item, course_meta)

        # Check content was normalized
        assert "\r\n" not in doc.page_content
        assert "\n\n\n" not in doc.page_content
        assert "Final Project" in doc.page_content
        assert "machine learning" in doc.page_content

        # Check metadata
        assert doc.metadata["source"] == "google_classroom"
        assert doc.metadata["course_id"] == "111"
        assert doc.metadata["course_name"] == "Integration Test Course"
        assert doc.metadata["content_type"] == "assignment"
        assert doc.metadata["due_date"] == "2024-03-15T23:59:00"
        assert doc.metadata["max_points"] == 100.0
        assert doc.metadata["item_id"] == "hw1"

    @pytest.mark.integration
    def test_full_announcement_pipeline(self) -> None:
        """Build an announcement Document and verify normalization."""
        course_meta = build_course_meta(SAMPLE_COURSE)
        raw_item = {
            "id": "ann1",
            "text": "Class is cancelled\r\ntomorrow\x00due to weather",
            "creationTime": "2024-02-01T08:00:00Z",
        }

        doc = build_from_announcement(raw_item, course_meta)

        assert "\r\n" not in doc.page_content
        assert "\x00" not in doc.page_content
        assert "cancelled" in doc.page_content
        assert doc.metadata["content_type"] == "announcement"

    @pytest.mark.integration
    def test_full_material_pipeline(self) -> None:
        """Build a material Document."""
        course_meta = build_course_meta(SAMPLE_COURSE)
        raw_item = {
            "id": "mat1",
            "title": "Week 5 Readings",
            "description": "Read chapters 7 and 8",
        }

        doc = build_from_material(raw_item, course_meta)

        assert "Week 5 Readings" in doc.page_content
        assert doc.metadata["content_type"] == "material"

    @pytest.mark.integration
    def test_attachment_builder(self) -> None:
        """Build an attachment Document with merged metadata."""
        course_meta = build_course_meta(SAMPLE_COURSE)
        parent = {"id": "hw1", "title": "Homework 1"}

        doc = build_from_attachment(
            file_id="file123",
            title="instructions.pdf",
            mime_type="text/plain",
            source_url="https://drive.google.com/file/d/file123",
            original_mime_type="application/pdf",
            parsed_text="Complete   \r\n\r\n\r\n exercises 1-5",
            parent_item=parent,
            course_meta=course_meta,
            content_type="assignment",
        )

        assert "\r\n" not in doc.page_content
        assert doc.metadata["content_type"] == "assignment_attachment"
        assert doc.metadata["file_id"] == "file123"
        assert doc.metadata["parent_title"] == "Homework 1"


# ---------------------------------------------------------------------------
# Integration: normalizer edge cases
# ---------------------------------------------------------------------------


class TestNormalizerIntegration:
    """Test normalizer with real-world messy content."""

    @pytest.mark.integration
    def test_real_world_content(self) -> None:
        """Simulate messy content from Google Classroom."""
        messy = (
            "  Assignment: Lab 3\r\n\r\n"
            "Complete the following:\r\n"
            "\x00\x00"
            "\n\n\n\n\n"
            "1. Build a neural network\n"
            "2. Train on MNIST\n"
            "\n\n\n\n"
            "Due by Friday  "
        )
        result = normalize(messy)

        assert "\r\n" not in result
        assert "\x00" not in result
        assert "\n\n\n" not in result
        assert result.startswith("Assignment")
        assert result.endswith("Friday")
        assert "1. Build a neural network" in result


# ---------------------------------------------------------------------------
# Integration: live API tests (skipped unless credentials exist)
# ---------------------------------------------------------------------------

_HAS_CREDENTIALS = os.path.exists("service_account.json") or os.path.exists(
    "token.json"
)


class _LiveAuthKwargs(TypedDict, total=False):
    service_account_file: str
    token_file: str


def _live_auth_kwargs() -> _LiveAuthKwargs:
    """Build auth kwargs for live integration tests.

    Prefer service account credentials when available. This avoids
    triggering OAuth client-secrets flow in environments that only
    provide ``service_account.json``.
    """
    if os.path.exists("service_account.json"):
        return {"service_account_file": "service_account.json"}
    if os.path.exists("token.json"):
        return {"token_file": "token.json"}
    return {}


@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_CREDENTIALS,
    reason="No Google credentials found (service_account.json or token.json)",
)
class TestLiveAPI:
    """Tests that hit the real Google Classroom API.

    These only run when valid credentials are present in the project
    root.  They are skipped in CI.

    To run:
        pytest tests/integration/ -m integration -v
    """

    def test_list_courses(self) -> None:
        """Should list at least one course."""
        loader = GoogleClassroomLoader(
            load_attachments=False,
            **_live_auth_kwargs(),
        )
        docs = loader.load()
        if len(docs) == 0:
            pytest.skip(
                "No accessible Classroom content for current credentials; "
                "grant course access to the service account or provide "
                "CLASSROOM_TEST_COURSE_ID for targeted testing."
            )
        assert all(isinstance(d, Document) for d in docs)
        assert all(d.metadata.get("source") == "google_classroom" for d in docs)

    def test_load_specific_course(self) -> None:
        """Load a specific course by ID.

        Set CLASSROOM_TEST_COURSE_ID env var to test.
        """
        course_id = os.environ.get("CLASSROOM_TEST_COURSE_ID")
        if not course_id:
            pytest.skip("CLASSROOM_TEST_COURSE_ID not set")

        loader = GoogleClassroomLoader(
            course_ids=[course_id],
            load_attachments=False,
            **_live_auth_kwargs(),
        )
        docs = loader.load()
        assert len(docs) > 0
        assert all(d.metadata.get("course_id") == course_id for d in docs)
