"""File parser registry for langchain-google-classroom."""

from __future__ import annotations

from typing import Dict, Optional, Type

from langchain_core.document_loaders import BaseBlobParser

from langchain_google_classroom.parsers.docx_parser import DocxParser
from langchain_google_classroom.parsers.image_parser import ImageParser
from langchain_google_classroom.parsers.pdf_parser import PDFParser
from langchain_google_classroom.parsers.text_parser import TextParser

__all__ = [
    "BaseBlobParser",
    "DocxParser",
    "ImageParser",
    "PDFParser",
    "TextParser",
    "get_parser",
]

# ---------------------------------------------------------------------------
# MIME type → parser class mapping
# ---------------------------------------------------------------------------

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

MIME_TYPE_PARSERS: Dict[str, Type[BaseBlobParser]] = {
    "application/pdf": PDFParser,
    _DOCX_MIME: DocxParser,
    "text/plain": TextParser,
    "text/csv": TextParser,
    "text/markdown": TextParser,
    "text/html": TextParser,
    "image/png": ImageParser,
    "image/jpg": ImageParser,
    "image/jpeg": ImageParser,
    "image/gif": ImageParser,
    "image/webp": ImageParser,
    "image/bmp": ImageParser,
}


def get_parser(mime_type: str) -> Optional[BaseBlobParser]:
    """Return a parser instance for *mime_type*, or ``None`` if unsupported.

    Every returned parser conforms to LangChain's
    :class:`~langchain_core.document_loaders.BaseBlobParser` interface and
    can be composed with any blob loader in the ecosystem.

    Args:
        mime_type: MIME type string (e.g. ``"application/pdf"``).

    Returns:
        A :class:`BaseBlobParser` subclass instance, or ``None``.
    """
    mime_type = mime_type.split(";")[0].strip().lower()
    parser_cls = MIME_TYPE_PARSERS.get(mime_type)

    if parser_cls is None:
        return None
    return parser_cls()
