"""LangChain Google Classroom integration.

Load courses, assignments, announcements, materials, student submissions,
rubrics, topics, rosters, and file attachments as structured LangChain
``Document`` objects.
"""

from __future__ import annotations

from importlib import metadata

from langchain_google_classroom.loader import GoogleClassroomLoader

# Parsers — exposed for users who want custom parser configurations
from langchain_google_classroom.parsers import (
    CSVParser,
    DocxParser,
    ImageParser,
    PDFParser,
    TextParser,
    get_parser,
)

try:
    __version__ = metadata.version("langchain-google-classroom")
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata

__all__ = [
    "GoogleClassroomLoader",
    "CSVParser",
    "DocxParser",
    "ImageParser",
    "PDFParser",
    "TextParser",
    "get_parser",
    "__version__",
]
