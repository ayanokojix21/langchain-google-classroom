# langchain-google-classroom

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-google-classroom?label=%20)](https://pypi.org/project/langchain-google-classroom/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-google-classroom)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-google-classroom)](https://pypistats.org/packages/langchain-google-classroom)
[![CI](https://github.com/ayanokojix21/langchain-google-classroom/actions/workflows/ci.yml/badge.svg)](https://github.com/ayanokojix21/langchain-google-classroom/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/langchain-google-classroom)](https://pypi.org/project/langchain-google-classroom/)

An integration package connecting **Google Classroom** and **LangChain**.

Load courses, assignments, announcements, materials, student submissions, rubrics,
topics, rosters, and file attachments as structured LangChain `Document` objects
— ready for RAG pipelines, AI teaching assistants, and educational analytics.

## Installation

```bash
pip install langchain-google-classroom
```

With optional parsers for PDF and DOCX attachments:

```bash
pip install "langchain-google-classroom[parsers]"
```

## Quick Start

```python
from langchain_google_classroom import GoogleClassroomLoader

loader = GoogleClassroomLoader(
    course_ids=["123456789"],
)
docs = loader.load()

for doc in docs:
    print(f"[{doc.metadata['content_type']}] {doc.metadata.get('title', '')}")
```

## What It Loads

| Content Type | Flag | Default |
|---|---|---|
| Assignments (courseWork) | `load_assignments` | `True` |
| Announcements | `load_announcements` | `True` |
| Course materials | `load_materials` | `True` |
| Drive attachments (PDF, DOCX, CSV, text, images) | `load_attachments` | `True` |
| YouTube and link attachments | `load_attachments` | `True` |
| Student submissions | `load_submissions` | `False` |
| Rubrics (per courseWork) | Always fetched with assignments | — |
| Topics | `load_topics` | `False` |
| Student and teacher roster | `load_roster` | `False` |

## Authentication

### OAuth (recommended for personal use)

```python
loader = GoogleClassroomLoader(
    client_secrets_file="credentials.json",
    token_file="token.json",
)
```

### Service Account (recommended for production)

```python
loader = GoogleClassroomLoader(
    service_account_file="service_account.json",
)
```

### Pre-built Credentials

```python
from google.oauth2 import service_account

creds = service_account.Credentials.from_service_account_file(
    "service_account.json",
    scopes=["https://www.googleapis.com/auth/classroom.courses.readonly"],
)
loader = GoogleClassroomLoader(credentials=creds)
```

> **Note:** Never commit credential files to version control. Use environment
> variables or secret managers in production.

## Drive Attachments

Drive files attached to classroom items are automatically downloaded and parsed:

```python
loader = GoogleClassroomLoader(
    course_ids=["123456789"],
    load_attachments=True,
    parse_attachments=True,
)
```

Google-native files are exported to parseable formats:

| Google Format | Exported As |
|---|---|
| Google Docs | DOCX |
| Google Slides | PDF |
| Google Sheets | CSV |

### Custom Parser

Override the built-in parser registry with any `BaseBlobParser` subclass:

```python
from langchain_community.document_loaders.parsers.pdf import PyMuPDFParser

loader = GoogleClassroomLoader(
    course_ids=["123456789"],
    file_parser_cls=PyMuPDFParser,
)
```

### File Size Limit

Skip oversized attachments to prevent memory issues:

```python
loader = GoogleClassroomLoader(
    max_file_size=10_000_000,  # 10 MB (default: 50 MB)
)
```

## Vision LLM

Extract and describe images embedded in PDFs and image attachments using any
vision-capable chat model:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

loader = GoogleClassroomLoader(
    course_ids=["123456789"],
    load_attachments=True,
    vision_model=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
)
docs = loader.load()
# PDF pages include: "[Image: chart.png]\nA bar chart showing student grades..."
```

## Student Submissions

Load submission state, grades, and student responses:

```python
loader = GoogleClassroomLoader(
    course_ids=["123456789"],
    load_submissions=True,
)
```

Each submission includes `state`, `late`, `assigned_grade`, `draft_grade`,
short-answer text, multiple-choice answers, and attachment file names.

## Topics and Roster

```python
loader = GoogleClassroomLoader(
    course_ids=["123456789"],
    load_topics=True,
    load_roster=True,
)
```

Required OAuth scopes are automatically added when these features are enabled.

## Async Support

```python
import asyncio
from langchain_google_classroom import GoogleClassroomLoader

async def main():
    loader = GoogleClassroomLoader(course_ids=["123456789"])
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)
    return docs

docs = asyncio.run(main())
```

## Document Structure

Every document includes structured metadata for filtering and retrieval:

```python
Document(
    page_content="Assignment: Homework 3\n\nComplete exercises 1-5...",
    metadata={
        "source": "google_classroom",
        "course_id": "12345",
        "course_name": "Machine Learning",
        "content_type": "assignment",
        "title": "Homework 3",
        "item_id": "67890",
        "created_time": "2024-01-15T10:00:00Z",
        "updated_time": "2024-01-15T10:00:00Z",
        "due_date": "2024-01-22T23:59:00",
        "max_points": 100.0,
        "alternate_link": "https://classroom.google.com/...",
    },
)
```

Content types: `assignment`, `announcement`, `material`, `submission`, `rubric`,
`topic`, `student`, `teacher`, `assignment_attachment`, `assignment_youtube`,
`assignment_link`.

## Configuration Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `course_ids` | `list[str]` | `None` | Specific course IDs to load (`None` = all accessible) |
| `load_assignments` | `bool` | `True` | Load courseWork items |
| `load_announcements` | `bool` | `True` | Load announcements |
| `load_materials` | `bool` | `True` | Load courseWorkMaterials |
| `load_attachments` | `bool` | `True` | Download and parse Drive file attachments |
| `parse_attachments` | `bool` | `True` | Parse files via `BaseBlobParser` (otherwise raw UTF-8) |
| `load_images` | `bool` | `False` | Process image MIME types |
| `load_submissions` | `bool` | `False` | Load student submissions |
| `load_topics` | `bool` | `False` | Load course topics |
| `load_roster` | `bool` | `False` | Load student and teacher profiles |
| `max_file_size` | `int` | `50000000` | Max attachment size in bytes (0 = no limit) |
| `vision_model` | `BaseChatModel` | `None` | Vision LLM for image understanding |
| `image_prompt` | `str` | `None` | Custom prompt for the vision model |
| `file_parser_cls` | `type` | `None` | Custom `BaseBlobParser` subclass for all attachments |
| `file_parser_kwargs` | `dict` | `{}` | Keyword arguments forwarded to `file_parser_cls` |
| `credentials` | `Credentials` | `None` | Pre-built Google credentials object |
| `service_account_file` | `str` | `None` | Path to service-account key JSON |
| `token_file` | `str` | `None` | Path to cached OAuth token JSON |
| `client_secrets_file` | `str` | `None` | Path to OAuth client-secrets JSON |
| `scopes` | `list[str]` | Read-only | API scopes (auto-extended when opt-in features are enabled) |

## Architecture

```
GoogleClassroomLoader (BaseLoader + BaseModel)
├── _utilities.py         — auth, retry/backoff, guard_import
├── _vision.py            — shared vision LLM helper
├── classroom_api.py      — paginated Classroom API fetcher
├── document_builder.py   — raw API responses → LangChain Document
├── drive_resolver.py     — Drive download/export + link extraction
├── normalizer.py         — text cleanup (Unicode NFC, whitespace)
└── parsers/
    ├── __init__.py       — MIME registry + get_parser()
    ├── pdf_parser.py     — pypdf + vision LLM
    ├── docx_parser.py    — python-docx + vision LLM
    ├── csv_parser.py     — header-aware CSV (Google Sheets exports)
    ├── text_parser.py    — built-in UTF-8
    └── image_parser.py   — vision LLM + base64 fallback
```

## Development

```bash
git clone https://github.com/ayanokojix21/langchain-google-classroom.git
cd langchain-google-classroom/libs/google-classroom
pip install -e ".[dev]"

# Tests
pytest tests/unit_tests/ -v --disable-socket --allow-unix-socket

# Lint and format
ruff check .
ruff format .
```

## License

MIT — see [LICENSE](LICENSE) for details.