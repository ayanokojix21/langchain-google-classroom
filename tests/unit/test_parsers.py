"""Unit tests for the file parser layer (BaseBlobParser interface)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents.base import Blob

from langchain_google_classroom.parsers import (
    DocxParser,
    ImageParser,
    PDFParser,
    TextParser,
    get_parser,
)

# ---------------------------------------------------------------------------
# Tests — TextParser
# ---------------------------------------------------------------------------


class TestTextParser:
    """Tests for TextParser (BaseBlobParser interface)."""

    def test_decode_utf8(self) -> None:
        blob = Blob(data=b"Hello, world!", mimetype="text/plain")
        parser = TextParser()
        docs = list(parser.lazy_parse(blob))
        assert len(docs) == 1
        assert docs[0].page_content == "Hello, world!"

    def test_decode_with_replacement(self) -> None:
        """Invalid bytes should be replaced, not cause an error."""
        blob = Blob(data=b"Hello \xff\xfe world", mimetype="text/plain")
        parser = TextParser()
        docs = list(parser.lazy_parse(blob))
        assert len(docs) == 1
        assert "Hello" in docs[0].page_content
        assert "world" in docs[0].page_content
        assert "\ufffd" in docs[0].page_content

    def test_empty_content(self) -> None:
        blob = Blob(data=b"", mimetype="text/plain")
        parser = TextParser()
        docs = list(parser.lazy_parse(blob))
        assert docs == []

    def test_multiline(self) -> None:
        blob = Blob(data=b"Line 1\nLine 2\nLine 3", mimetype="text/plain")
        parser = TextParser()
        docs = list(parser.lazy_parse(blob))
        assert len(docs) == 1
        assert docs[0].page_content == "Line 1\nLine 2\nLine 3"

    def test_source_metadata(self) -> None:
        blob = Blob(
            data=b"test",
            mimetype="text/plain",
            path="https://example.com/file.txt",
        )
        parser = TextParser()
        docs = list(parser.lazy_parse(blob))
        assert docs[0].metadata["source"] == "https://example.com/file.txt"


# ---------------------------------------------------------------------------
# Tests — PDFParser
# ---------------------------------------------------------------------------


class TestPDFParser:
    """Tests for PDFParser (pypdf mocked, BaseBlobParser interface)."""

    def test_extracts_text_per_page(self) -> None:
        """Each page yields a separate Document."""
        mock_page_1 = MagicMock()
        mock_page_1.extract_text.return_value = "Page 1 content"
        mock_page_2 = MagicMock()
        mock_page_2.extract_text.return_value = "Page 2 content"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page_1, mock_page_2]

        with patch(
            "langchain_google_classroom.parsers.pdf_parser.guard_import"
        ) as mock_guard:
            mock_module = MagicMock()
            mock_module.PdfReader.return_value = mock_reader
            mock_guard.return_value = mock_module

            parser = PDFParser()
            blob = Blob(data=b"fake pdf bytes", mimetype="application/pdf")
            docs = list(parser.lazy_parse(blob))

        assert len(docs) == 2
        assert docs[0].page_content == "Page 1 content"
        assert docs[0].metadata["page"] == 1
        assert docs[1].page_content == "Page 2 content"
        assert docs[1].metadata["page"] == 2

    def test_empty_pdf(self) -> None:
        """PDF with no pages should yield nothing."""
        mock_reader = MagicMock()
        mock_reader.pages = []

        with patch(
            "langchain_google_classroom.parsers.pdf_parser.guard_import"
        ) as mock_guard:
            mock_module = MagicMock()
            mock_module.PdfReader.return_value = mock_reader
            mock_guard.return_value = mock_module

            parser = PDFParser()
            blob = Blob(data=b"fake pdf bytes", mimetype="application/pdf")
            docs = list(parser.lazy_parse(blob))

        assert docs == []

    def test_page_with_no_text(self) -> None:
        """Pages returning None for extract_text should be skipped."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch(
            "langchain_google_classroom.parsers.pdf_parser.guard_import"
        ) as mock_guard:
            mock_module = MagicMock()
            mock_module.PdfReader.return_value = mock_reader
            mock_guard.return_value = mock_module

            parser = PDFParser()
            blob = Blob(data=b"fake pdf bytes", mimetype="application/pdf")
            docs = list(parser.lazy_parse(blob))

        assert docs == []


# ---------------------------------------------------------------------------
# Tests — DocxParser
# ---------------------------------------------------------------------------


class TestDocxParser:
    """Tests for DocxParser (python-docx mocked, BaseBlobParser interface)."""

    def test_extracts_paragraphs(self) -> None:
        """All non-empty paragraphs as a single Document."""
        mock_para_1 = MagicMock()
        mock_para_1.text = "First paragraph"
        mock_para_2 = MagicMock()
        mock_para_2.text = "Second paragraph"
        mock_para_empty = MagicMock()
        mock_para_empty.text = "   "

        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para_1, mock_para_empty, mock_para_2]

        with patch(
            "langchain_google_classroom.parsers.docx_parser.guard_import"
        ) as mock_guard:
            mock_module = MagicMock()
            mock_module.Document.return_value = mock_doc
            mock_guard.return_value = mock_module

            parser = DocxParser()
            blob = Blob(
                data=b"fake docx bytes",
                mimetype=(
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                ),
            )
            docs = list(parser.lazy_parse(blob))

        assert len(docs) == 1
        assert docs[0].page_content == "First paragraph\n\nSecond paragraph"

    def test_empty_document(self) -> None:
        """Document with no paragraphs should yield nothing."""
        mock_doc = MagicMock()
        mock_doc.paragraphs = []

        with patch(
            "langchain_google_classroom.parsers.docx_parser.guard_import"
        ) as mock_guard:
            mock_module = MagicMock()
            mock_module.Document.return_value = mock_doc
            mock_guard.return_value = mock_module

            parser = DocxParser()
            blob = Blob(
                data=b"fake docx bytes",
                mimetype=(
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                ),
            )
            docs = list(parser.lazy_parse(blob))

        assert docs == []

    def test_docx_images_with_vision(self) -> None:
        """Embedded DOCX images are described when a vision model is set."""
        mock_para = MagicMock()
        mock_para.text = "Main text"

        mock_image_part = MagicMock()
        mock_image_part.blob = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        mock_image_part.partname = "/word/media/image1.png"

        mock_rel = MagicMock()
        mock_rel.reltype = (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
        )
        mock_rel.target_part = mock_image_part

        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para]
        mock_doc.part.rels = {"rId1": mock_rel}

        with patch(
            "langchain_google_classroom.parsers.docx_parser.guard_import"
        ) as mock_guard:
            mock_module = MagicMock()
            mock_module.Document.return_value = mock_doc
            mock_guard.return_value = mock_module

            mock_vision = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "A chart with student scores."
            mock_vision.invoke.return_value = mock_response

            parser = DocxParser(vision_model=mock_vision)
            blob = Blob(
                data=b"fake docx bytes",
                mimetype=(
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                ),
            )
            docs = list(parser.lazy_parse(blob))

        assert len(docs) == 1
        assert "Main text" in docs[0].page_content
        assert "[Image: /word/media/image1.png]" in docs[0].page_content
        assert "A chart with student scores." in docs[0].page_content
        mock_vision.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# Tests — Parser registry
# ---------------------------------------------------------------------------


class TestGetParser:
    """Tests for the get_parser registry function."""

    def test_pdf_mime_returns_pdf_parser(self) -> None:
        parser = get_parser("application/pdf")
        assert isinstance(parser, PDFParser)

    def test_docx_mime_returns_docx_parser(self) -> None:
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        parser = get_parser(mime)
        assert isinstance(parser, DocxParser)

    def test_text_plain_returns_text_parser(self) -> None:
        parser = get_parser("text/plain")
        assert isinstance(parser, TextParser)

    def test_text_csv_returns_text_parser(self) -> None:
        parser = get_parser("text/csv")
        assert isinstance(parser, TextParser)

    def test_text_html_returns_text_parser(self) -> None:
        parser = get_parser("text/html")
        assert isinstance(parser, TextParser)

    def test_unknown_mime_returns_none(self) -> None:
        assert get_parser("audio/mp3") is None

    def test_unsupported_mime_returns_none(self) -> None:
        assert get_parser("application/zip") is None

    def test_is_base_blob_parser(self) -> None:
        """All parsers should be BaseBlobParser subclasses."""
        for mime in ["application/pdf", "text/plain", "image/png"]:
            parser = get_parser(mime)
            assert isinstance(parser, BaseBlobParser)

    def test_image_png_returns_image_parser(self) -> None:
        parser = get_parser("image/png")
        assert isinstance(parser, ImageParser)

    def test_image_jpeg_returns_image_parser(self) -> None:
        parser = get_parser("image/jpeg")
        assert isinstance(parser, ImageParser)


# ---------------------------------------------------------------------------
# Tests — ImageParser
# ---------------------------------------------------------------------------


class TestImageParser:
    """Tests for ImageParser (BaseBlobParser interface)."""

    def test_no_vision_yields_base64_metadata(self) -> None:
        """Without vision model, base64 is stored in metadata."""
        blob = Blob(
            data=b"\x89PNG fake image data",
            mimetype="image/png",
            path="https://drive.google.com/photo.png",
        )
        parser = ImageParser()
        docs = list(parser.lazy_parse(blob))

        assert len(docs) == 1
        assert "[Image:" in docs[0].page_content
        assert "image_base64" in docs[0].metadata
        assert docs[0].metadata["mime_type"] == "image/png"

    def test_with_vision_model(self) -> None:
        """With vision model, description replaces base64."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "A bar chart showing grades."
        mock_model.invoke.return_value = mock_response

        blob = Blob(
            data=b"\x89PNG fake image data",
            mimetype="image/png",
        )
        parser = ImageParser(vision_model=mock_model)
        docs = list(parser.lazy_parse(blob))

        assert len(docs) == 1
        assert "A bar chart showing grades." in docs[0].page_content
        assert "image_base64" not in docs[0].metadata
        mock_model.invoke.assert_called_once()

    def test_vision_error_graceful(self) -> None:
        """Vision failure yields placeholder, not crash."""
        mock_model = MagicMock()
        mock_model.invoke.side_effect = RuntimeError("API error")

        blob = Blob(
            data=b"\x89PNG fake image data",
            mimetype="image/png",
        )
        parser = ImageParser(vision_model=mock_model)
        docs = list(parser.lazy_parse(blob))

        assert len(docs) == 1
        assert "[Image:" in docs[0].page_content

    def test_empty_image(self) -> None:
        """Empty blob should yield nothing."""
        blob = Blob(data=b"", mimetype="image/png")
        parser = ImageParser()
        docs = list(parser.lazy_parse(blob))
        assert docs == []


# ---------------------------------------------------------------------------
# Tests — PDFParser with vision model
# ---------------------------------------------------------------------------


class TestPDFParserVision:
    """Tests for PDFParser vision model integration."""

    def test_no_vision_skips_images(self) -> None:
        """Without vision model, images are not processed."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page text"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch(
            "langchain_google_classroom.parsers.pdf_parser.guard_import"
        ) as mock_guard:
            mock_module = MagicMock()
            mock_module.PdfReader.return_value = mock_reader
            mock_guard.return_value = mock_module

            parser = PDFParser()
            blob = Blob(data=b"fake pdf", mimetype="application/pdf")
            docs = list(parser.lazy_parse(blob))

        assert len(docs) == 1
        assert docs[0].page_content == "Page text"

    def test_vision_describes_embedded_images(self) -> None:
        """With vision model, embedded images are described."""
        mock_img = MagicMock()
        mock_img.data = b"\x89PNG fake"
        mock_img.name = "chart.png"

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Statistics"
        mock_page.images = [mock_img]

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "A pie chart of grades."
        mock_model.invoke.return_value = mock_response

        with patch(
            "langchain_google_classroom.parsers.pdf_parser.guard_import"
        ) as mock_guard:
            mock_module = MagicMock()
            mock_module.PdfReader.return_value = mock_reader
            mock_guard.return_value = mock_module

            parser = PDFParser(vision_model=mock_model)
            blob = Blob(data=b"fake pdf", mimetype="application/pdf")
            docs = list(parser.lazy_parse(blob))

        assert len(docs) == 1
        assert "Statistics" in docs[0].page_content
        assert "[Image: chart.png]" in docs[0].page_content
        assert "A pie chart of grades." in docs[0].page_content
        mock_model.invoke.assert_called_once()

    def test_vision_error_still_yields_text(self) -> None:
        """Vision failure should not block text extraction."""
        mock_img = MagicMock()
        mock_img.data = b"\x89PNG fake"
        mock_img.name = "broken.png"

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page text"
        mock_page.images = [mock_img]

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        mock_model = MagicMock()
        mock_model.invoke.side_effect = RuntimeError("API down")

        with patch(
            "langchain_google_classroom.parsers.pdf_parser.guard_import"
        ) as mock_guard:
            mock_module = MagicMock()
            mock_module.PdfReader.return_value = mock_reader
            mock_guard.return_value = mock_module

            parser = PDFParser(vision_model=mock_model)
            blob = Blob(data=b"fake pdf", mimetype="application/pdf")
            docs = list(parser.lazy_parse(blob))

        assert len(docs) == 1
        assert "Page text" in docs[0].page_content
