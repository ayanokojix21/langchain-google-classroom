"""Unit tests for the content normalizer."""

from __future__ import annotations

import pytest

from langchain_google_classroom.normalizer import normalize


class TestNormalize:
    """Tests for the normalize function."""

    def test_empty_string(self) -> None:
        assert normalize("") == ""

    def test_strip_whitespace(self) -> None:
        assert normalize("  hello  ") == "hello"

    def test_collapse_newlines(self) -> None:
        assert normalize("a\n\n\n\nb") == "a\n\nb"

    def test_windows_line_endings(self) -> None:
        assert normalize("a\r\nb\r\n") == "a\nb"

    def test_null_bytes_removed(self) -> None:
        assert normalize("hello\x00world") == "helloworld"

    def test_unicode_normalisation(self) -> None:
        # é as e + combining acute (NFD) should become single char (NFC)
        nfd = "e\u0301"
        result = normalize(nfd)
        assert result == "\u00e9"

    def test_preserves_normal_text(self) -> None:
        text = "This is a normal paragraph.\n\nThis is another."
        assert normalize(text) == text

    def test_mixed_issues(self) -> None:
        text = "  hello\x00\r\n\r\n\r\n\r\nworld  "
        result = normalize(text)
        assert result == "hello\n\nworld"
