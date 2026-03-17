# 🎓 langchain-google-classroom

[![CI](https://github.com/ayanokojix21/langchain-google-classroom/actions/workflows/ci.yml/badge.svg)](https://github.com/ayanokojix21/langchain-google-classroom/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/langchain-google-classroom.svg)](https://pypi.org/project/langchain-google-classroom/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-google-classroom.svg)](https://pypi.org/project/langchain-google-classroom/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Status: Community integration package (not officially listed by LangChain yet).

A community package for loading Google Classroom content — assignments, announcements, course materials, and Drive attachments — as `Document` objects for RAG pipelines, semantic search, AI teaching assistants, and course chatbots.

Compatible with **LangChain Document Loaders**.

## 🤔 What is this?

Google Classroom data is difficult to integrate directly into LLM workflows.
This package converts Classroom content into LangChain `Document` objects, making it easier to build:

- AI teaching assistants
- Course chatbots
- Semantic search over coursework
- Automated grading helpers

## 📖 Documentation

- LangChain docs: https://docs.langchain.com/oss/python/langchain/overview
- LangChain integrations overview: https://docs.langchain.com/oss/python/integrations/providers/overview
- Contributing to LangChain docs: https://docs.langchain.com/oss/python/contributing/overview

## ✨ Features

- **Full Classroom coverage** — assignments, announcements, and course materials
- **Drive attachments** — auto-download and parse PDF, DOCX, text, CSV, HTML files
- **Vision LLM image understanding** — embedded PDF images described by Gemini/GPT-4V
- **Pluggable parsers** — bring your own `BaseBlobParser` (PyMuPDF, Unstructured, etc.)
- **Retry/backoff** — exponential backoff with jitter on rate-limited API calls
- **Flexible auth** — service accounts, OAuth, cached tokens, or pre-built credentials
- **Rich metadata** — course info, timestamps, due dates, links on every Document
- **Lazy loading** — memory-efficient streaming via `lazy_load()`

## 📦 Installation

Requires Python >=3.10.

```bash
pip install langchain-google-classroom
```

With file attachment parsing (PDF, DOCX):

```bash
pip install "langchain-google-classroom[parsers]"
```

## 🚀 Quickstart

```python
from langchain_google_classroom import GoogleClassroomLoader

# Load all accessible courses
loader = GoogleClassroomLoader()
docs = loader.load()  # eager loading

for doc in docs:
    print(doc.metadata["content_type"], "—", doc.metadata["title"])
    print(doc.page_content[:200])
    print()

# Lazy loading (stream documents one by one)
for doc in loader.lazy_load():
    print(doc.metadata["content_type"], "—", doc.metadata["title"])
```

See [examples/](examples/) for more usage examples.

Sample output:

```text
assignment — Homework 3
announcement — Exam postponed
material — Lecture 4 Slides
```

## 🧠 RAG Example (Optional)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_classroom import GoogleClassroomLoader

loader = GoogleClassroomLoader()
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
```

Install optional dependencies for this example as needed (`langchain-text-splitters`, `langchain-openai`, `faiss-cpu`).

## 🔐 Setup Credentials

Google Classroom APIs require authentication. Use one of the following methods:

### 1. OAuth User Credentials (Recommended)

This is the easiest way to start. When you run this the first time, your browser will open asking you to log in with your Google Account, and it will generate a `token.json` for all future requests.

```python
loader = GoogleClassroomLoader(
    client_secrets_file="credentials.json",
    token_file="token.json",
)
```

### 2. Service Account

Service accounts do not require human interaction. However, please note that Service Accounts act as "bot users" and cannot see your personal Google Classroom courses unless your Google Workspace Administrator explicitly grants them "Domain-Wide Delegation" for classroom scopes.

```python
loader = GoogleClassroomLoader(
    service_account_file="service_account.json",
)
```

### 3. Pre-built Credentials Object

```python
from google.oauth2 import service_account

creds = service_account.Credentials.from_service_account_file(
    "service_account.json",
    scopes=["https://www.googleapis.com/auth/classroom.courses.readonly"],
)
loader = GoogleClassroomLoader(credentials=creds)
```

Credential safety:

- Never commit `credentials.json`, `token.json`, or `service_account.json`.
- Use GitHub Actions Secrets for CI integration tests.

## 📎 Attachments & File Parsing

```python
loader = GoogleClassroomLoader(
    course_ids=["123456789"],
    load_attachments=True,      # Download Drive files
    parse_attachments=True,     # Parse with BaseBlobParser
)
docs = loader.load()
# Yields: assignment docs + parsed PDF/DOCX/text attachment docs
```

### Custom Parser

```python
from langchain_community.document_loaders.parsers.pdf import PyMuPDFParser

loader = GoogleClassroomLoader(
    course_ids=["123456789"],
    file_parser_cls=PyMuPDFParser,
)
```

## 🖼️ Vision LLM — Image Understanding

Extract and describe images embedded in PDFs using any vision-capable LLM:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

loader = GoogleClassroomLoader(
    course_ids=["123456789"],
    load_attachments=True,
    vision_model=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
)
docs = loader.load()
# PDF pages now include: "[Image: chart.png]\nA bar chart showing student grades..."
```

## 🎯 Selective Loading

```python
loader = GoogleClassroomLoader(
    course_ids=["123456789"],
    load_assignments=True,
    load_announcements=False,
    load_materials=False,
    load_attachments=False,
)
```

## 📄 Document Structure

Each document includes rich metadata:

```python
Document(
    page_content="Assignment: Homework 3\n\nComplete exercises 1-5...",
    metadata={
        "source": "google_classroom",
        "course_id": "12345",
        "course_name": "Machine Learning",
        "content_type": "assignment",        # or "announcement", "material", "assignment_attachment"
        "title": "Homework 3",
        "item_id": "67890",
        "created_time": "2024-01-15T10:00:00Z",
        "updated_time": "2024-01-15T10:00:00Z",
        "due_date": "2024-01-22T23:59:00",   # assignments only
        "max_points": 100,                    # assignments only
        "alternate_link": "https://classroom.google.com/...",
    }
)
```

## ⚙️ Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `course_ids` | `list[str]` | `None` | Specific course IDs (`None` = all accessible) |
| `load_assignments` | `bool` | `True` | Load courseWork items |
| `load_announcements` | `bool` | `True` | Load announcements |
| `load_materials` | `bool` | `True` | Load courseWorkMaterials |
| `load_attachments` | `bool` | `True` | Download and process Drive attachments |
| `parse_attachments` | `bool` | `True` | Parse files with BaseBlobParser |
| `load_images` | `bool` | `False` | Process image MIME types |
| `vision_model` | `BaseChatModel` | `None` | Vision LLM for image understanding |
| `image_prompt` | `str` | `None` | Custom prompt for vision model |
| `file_parser_cls` | `type[BaseBlobParser]` | `None` | Custom parser for all attachments |
| `file_parser_kwargs` | `dict` | `None` | kwargs for custom parser |
| `credentials` | `Credentials` | `None` | Pre-built Google credentials |
| `service_account_file` | `str` | `None` | Service account key JSON path |
| `token_file` | `str` | `None` | Cached OAuth token path |
| `client_secrets_file` | `str` | `None` | OAuth client secrets path |
| `scopes` | `list[str]` | Read-only | API scopes to request |

## 🏗️ Architecture

```
GoogleClassroomLoader (BaseLoader)
├── _utilities.py         — auth, retry/backoff, guard_import
├── classroom_api.py      — paginated Classroom API fetcher
├── document_builder.py   — raw API → LangChain Document
├── drive_resolver.py     — Drive download/export
├── normalizer.py         — text cleanup (Unicode NFC, whitespace)
└── parsers/
    ├── __init__.py       — MIME registry + get_parser()
    ├── pdf_parser.py     — pypdf + vision LLM
    ├── docx_parser.py    — python-docx
    ├── text_parser.py    — built-in UTF-8
    └── image_parser.py   — vision LLM + base64 fallback
```

## 🧪 Development

```bash
# Clone and install
git clone https://github.com/ayanokojix21/langchain-google-classroom.git
cd langchain-google-classroom
python -m pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format + lint
ruff format .
ruff check .
```

## 📝 License

MIT — see [LICENSE](LICENSE) for details.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.