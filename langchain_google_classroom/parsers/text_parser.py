"""Plain text file parser (BaseBlobParser interface)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_core.documents.base import Blob


class TextParser(BaseBlobParser):
    """Parse plain text, CSV, and HTML files.

    Conforms to LangChain's :class:`BaseBlobParser` interface.
    """

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Decode raw bytes as UTF-8 and yield a single ``Document``.

        Invalid byte sequences are replaced with the Unicode replacement
        character to prevent decoding errors.

        Args:
            blob: A LangChain ``Blob`` containing text bytes.

        Yields:
            A single ``Document`` with the decoded text.
        """
        text = blob.as_bytes().decode("utf-8", errors="replace")
        if text.strip():
            yield Document(
                page_content=text,
                metadata={"source": blob.source or ""},
            )
