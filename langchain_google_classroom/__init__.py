from __future__ import annotations

from langchain_google_classroom.drive_resolver import DriveAttachmentResolver
from langchain_google_classroom.loader import GoogleClassroomLoader
from langchain_google_classroom.parsers import get_parser

__all__ = [
    "DriveAttachmentResolver",
    "GoogleClassroomLoader",
    "get_parser",
    "__version__",
]

from importlib import metadata

try:
    __version__ = metadata.version("langchain-google-classroom")
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata
