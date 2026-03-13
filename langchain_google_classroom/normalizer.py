"""Content normalizer for cleaning up raw text from Google Classroom."""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """Clean and normalize raw text content.

    The normalizer performs the following transformations:

    * Replace ``\\r\\n`` with ``\\n``.
    * Remove null bytes.
    * Normalise Unicode to NFC form.
    * Collapse runs of more than two consecutive newlines.
    * Strip leading/trailing whitespace.

    Args:
        text: Raw text to normalise.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null bytes
    text = text.replace("\x00", "")

    # Unicode NFC normalisation
    text = unicodedata.normalize("NFC", text)

    # Collapse excessive blank lines (keep at most one blank line)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text
