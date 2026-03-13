from langchain_google_classroom.loader import GoogleClassroomLoader

__all__ = ["GoogleClassroomLoader"]

from importlib import metadata

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata