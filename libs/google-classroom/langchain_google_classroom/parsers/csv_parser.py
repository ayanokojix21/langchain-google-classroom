"""CSV file parser for Google Sheets exports (BaseBlobParser interface).

Google Sheets attached to Classroom items are exported as CSV via the
Drive API.  This parser produces structured ``Document`` objects with
header-aware content and row-count metadata — a significant improvement
over the previous ``TextParser`` fallback, which treated CSV as raw text.
"""

from __future__ import annotations

import csv
import io
import logging
from typing import TYPE_CHECKING, Any, Iterator

from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_core.documents.base import Blob

logger = logging.getLogger(__name__)


class CSVParser(BaseBlobParser):
    """Parse CSV files into ``Document`` objects.

    Produces a single ``Document`` with a readable table representation.
    Headers are detected from the first row and each data row is
    formatted as ``Header: Value`` pairs for optimal LLM comprehension.

    When the CSV has fewer than ``max_inline_rows`` rows, the full
    content is inlined.  Larger files include a summary header and the
    first ``max_inline_rows`` rows.

    Conforms to LangChain's :class:`BaseBlobParser` interface.

    !!! example "Basic Usage"

        ```python
        from langchain_google_classroom.parsers import CSVParser

        parser = CSVParser()
        docs = list(parser.lazy_parse(blob))
        ```
    """

    def __init__(
        self,
        *,
        max_inline_rows: int = 500,
        include_row_numbers: bool = True,
    ) -> None:
        """Initialise the CSV parser.

        Args:
            max_inline_rows: Maximum number of data rows to include in
                the document content.  Larger sheets are truncated with
                a summary note.
            include_row_numbers: Whether to prefix each row with its
                number (1-indexed, header excluded).
        """
        self.max_inline_rows = max_inline_rows
        self.include_row_numbers = include_row_numbers

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Parse a CSV blob into a single Document.

        Args:
            blob: A LangChain ``Blob`` containing CSV text bytes.

        Yields:
            One ``Document`` with structured row content and
            metadata including ``row_count``, ``column_count``, and
            ``headers``.
        """
        text = blob.as_bytes().decode("utf-8", errors="replace")

        # Strip BOM if present (common in Google Sheets CSV exports)
        if text.startswith("\ufeff"):
            text = text[1:]

        if not text.strip():
            return

        reader = csv.reader(io.StringIO(text))
        rows = list(reader)

        if not rows:
            return

        # Remove fully empty rows (common Google Sheets quirk)
        rows = [row for row in rows if any(cell.strip() for cell in row)]

        if not rows:
            return

        headers = rows[0]
        data_rows = rows[1:]
        total_rows = len(data_rows)
        truncated = total_rows > self.max_inline_rows

        # Build page content
        parts: list[str] = []

        if truncated:
            parts.append(
                f"Spreadsheet ({total_rows} rows, showing first "
                f"{self.max_inline_rows}):"
            )
        else:
            parts.append(f"Spreadsheet ({total_rows} rows):")

        parts.append(f"Columns: {', '.join(headers)}")
        parts.append("")

        display_rows = data_rows[: self.max_inline_rows]
        for i, row in enumerate(display_rows, 1):
            entries: list[str] = []
            for header, value in zip(headers, row):
                value = value.strip()
                if value:
                    entries.append(f"{header}: {value}")
            if entries:
                if self.include_row_numbers:
                    parts.append(f"Row {i}: {' | '.join(entries)}")
                else:
                    parts.append(" | ".join(entries))

        if truncated:
            parts.append(
                f"\n... ({total_rows - self.max_inline_rows} more rows omitted)"
            )

        page_content = "\n".join(parts)

        metadata: dict[str, Any] = {
            "source": blob.source or "",
            "mime_type": blob.mimetype or "text/csv",
            "row_count": total_rows,
            "column_count": len(headers),
            "headers": headers,
        }

        if truncated:
            metadata["truncated"] = True

        yield Document(page_content=page_content, metadata=metadata)
