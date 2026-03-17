# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-13

### Added

- **`GoogleClassroomLoader`** — LangChain `BaseLoader` for Google Classroom.
  - Loads assignments (courseWork), announcements, and course materials.
  - Selective loading via `load_assignments`, `load_announcements`, `load_materials` flags.
  - Multi-course support with `course_ids` filter.
- **Drive attachment resolution** via `DriveAttachmentResolver`.
  - Google-native files (Docs, Slides, Sheets) exported as DOCX / PDF / CSV.
  - Binary files (PDF, DOCX, etc.) downloaded via streaming.
  - Controlled by `load_attachments` and `parse_attachments` flags.
- **File parser layer** using LangChain's `BaseBlobParser` + `Blob` interface.
  - `PDFParser` (pypdf), `DocxParser` (python-docx), `TextParser` (built-in),
    `ImageParser` (vision LLM).
  - MIME-type registry with `get_parser()` factory function.
  - `file_parser_cls` param for user-pluggable parsers (e.g. `PyMuPDFParser`).
- **Vision LLM image understanding** for images embedded in PDFs and standalone
  image attachments.  Pass a `vision_model` (e.g. Gemini, GPT-4V) to get
  automatic image understanding context appended to page text.
- **Retry / backoff** via `execute_with_retry()` on all Google API calls.
  - Exponential backoff with jitter on HTTP 429, 500, 503.
- **Authentication** support for service accounts, cached OAuth tokens, and interactive
  OAuth flow.
- **Text normalizer** — NFC, line ending cleanup, null byte removal.
- **`Document` metadata** — rich metadata including course info, timestamps, due dates,
  links, and parent item references for attachments.
- **Project packaging** — `pyproject.toml` with hatchling, optional dependency groups
  (`[parsers]`, `[test]`, `[lint]`, `[typing]`, `[dev]`), PEP 561 `py.typed` marker.
- **Comprehensive test coverage** across modules, including unit and integration tests.

[Unreleased]: https://github.com/ayanokojix21/langchain-google-classroom/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ayanokojix21/langchain-google-classroom/releases/tag/v0.1.0