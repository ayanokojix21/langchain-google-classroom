# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-14

### Added

- **Student submissions** — `load_submissions=True` yields submission state, grades,
  short-answer and multiple-choice responses, and attachment file names.
- **Rubrics** — rubrics are automatically fetched for every courseWork item and
  rendered as structured text with criteria, levels, and point values.
- **Topics** — `load_topics=True` yields course topic names and IDs.
- **Roster** — `load_roster=True` yields student and teacher profiles with names and
  emails.
- **YouTube and link attachments** — YouTube videos and external links on classroom
  items are captured as `Document` objects with structured metadata (video ID, URL,
  thumbnail).
- **CSV parser** (`CSVParser`) — Google Sheets exports are parsed with header-aware
  formatting (`Header: Value` pairs), row-count metadata, BOM handling, and
  configurable truncation for large sheets.
- **File size guard** — `max_file_size` parameter (default 50 MB) skips oversized
  attachments with a warning log. Google-native files (size reported as 0) are always
  allowed through.
- **Configurable retry delay** — `base_delay` parameter on `execute_with_retry()`
  allows callers to tune exponential backoff timing.
- **Shared vision helper** (`_vision.py`) — centralised `describe_image()` function
  used by `PDFParser`, `DocxParser`, and `ImageParser`, eliminating duplicated
  vision-LLM integration code.
- **Automatic scope injection** — enabling `load_submissions`, `load_roster`, or
  `load_topics` automatically appends the required OAuth scopes via a Pydantic
  `model_validator`.
- **`model_config`** — `ConfigDict(arbitrary_types_allowed=True)` on the loader for
  proper Pydantic v2 compatibility with non-serialisable credential objects.
- **Credential exclusion** — the `credentials` field is marked `exclude=True` to
  prevent sensitive objects from appearing in `model_dump()` output.
- **`PrivateAttr` parser cache** — `_parser_cache` uses Pydantic `PrivateAttr` for
  correct per-instance isolation.

### Changed

- **Repository restructure** — source code moved under `libs/google-classroom/` to
  match the `langchain-google` monorepo convention.
- **Pydantic v2 migration** — the loader class now inherits from both `BaseLoader` and
  `BaseModel` with proper `Field()` declarations, replacing the legacy `__init__`
  signature.
- **Parser interface** — all parsers (`PDFParser`, `DocxParser`, `ImageParser`,
  `TextParser`, `CSVParser`) conform to `BaseBlobParser` with the `lazy_parse(blob)`
  interface for full LangChain ecosystem composability.
- **`ClassroomAPIFetcher`** — extracted from the loader into a dedicated module with
  paginated generators for all nine Classroom API endpoints.
- **`DocumentBuilder`** — extracted from the loader into a dedicated module with typed
  builder functions for each content type.
- **Import organisation** — all Google SDK imports use `guard_import()` for lazy
  loading with helpful `pip_name` error messages.

### Fixed

- Removed unused `base64` import in `docx_parser.py`.
- Removed unnecessary `str()` wrappers around `blob.source` (already `str`).
- Added explicit `str()` casts for `dict.get()` returns passed to `NamedTuple` fields
  in `drive_resolver.py` to satisfy type checkers.
- Fixed stale `TextParser` docstring that still referenced CSV handling.
- Removed unused `Sequence` and `Type` imports from `loader.py`.

## [0.1.2] - 2026-03-17

### Fixed

- Re-released to update the `README.md` and force-refresh the PyPI cached rendering
  of the Python versions and Version badge.

## [0.1.1] - 2026-03-17

### Fixed

- Resolved typing anomalies (`dict[str, str]` vs `TypedDict`) in the integration
  tests which were overlooked by the previously scoped `mypy` CI command.
- Updated `.github/workflows/ci.yml` to strictly type-check all tests using `mypy .`.

## [0.1.0] - 2026-03-13

### Added

- **`GoogleClassroomLoader`** — LangChain `BaseLoader` for Google Classroom.
  - Loads assignments (courseWork), announcements, and course materials.
  - Selective loading via `load_assignments`, `load_announcements`, `load_materials`.
  - Multi-course support with `course_ids` filter.
- **Drive attachment resolution** via `DriveAttachmentResolver`.
  - Google-native files (Docs, Slides, Sheets) exported as DOCX / PDF / CSV.
  - Binary files (PDF, DOCX, etc.) downloaded via streaming.
  - Controlled by `load_attachments` and `parse_attachments` flags.
- **File parser layer** using LangChain's `BaseBlobParser` + `Blob` interface.
  - `PDFParser` (pypdf), `DocxParser` (python-docx), `TextParser` (built-in),
    `ImageParser` (vision LLM).
  - MIME-type registry with `get_parser()` factory function.
  - `file_parser_cls` parameter for user-pluggable parsers.
- **Vision LLM image understanding** for images embedded in PDFs and standalone
  image attachments.
- **Retry / backoff** via `execute_with_retry()` on all Google API calls.
- **Authentication** support for service accounts, cached OAuth tokens, and interactive
  OAuth flow.
- **Text normaliser** — NFC, line ending cleanup, null byte removal.
- **Rich metadata** — course info, timestamps, due dates, links, and parent item
  references on every `Document`.
- **Project packaging** — `pyproject.toml` with hatchling, optional dependency groups
  (`[parsers]`, `[test]`, `[lint]`, `[typing]`, `[dev]`), PEP 561 `py.typed` marker.

[Unreleased]: https://github.com/ayanokojix21/langchain-google-classroom/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ayanokojix21/langchain-google-classroom/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/ayanokojix21/langchain-google-classroom/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ayanokojix21/langchain-google-classroom/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ayanokojix21/langchain-google-classroom/releases/tag/v0.1.0