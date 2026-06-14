# langchain-google-classroom

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-google-classroom?label=%20)](https://pypi.org/project/langchain-google-classroom/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-google-classroom)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-google-classroom)](https://pypistats.org/packages/langchain-google-classroom)
[![CI](https://github.com/ayanokojix21/langchain-google-classroom/actions/workflows/ci.yml/badge.svg)](https://github.com/ayanokojix21/langchain-google-classroom/actions/workflows/ci.yml)

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

# OAuth (opens browser on first run)
loader = GoogleClassroomLoader(
    course_ids=["123456789"],
)
docs = loader.load()

for doc in docs:
    print(f"[{doc.metadata['content_type']}] {doc.metadata.get('title', '')}")
```

### Service Account

```python
loader = GoogleClassroomLoader(
    service_account_file="service_account.json",
)
```

### With Attachments and Vision LLM

```python
from langchain_google_genai import ChatGoogleGenerativeAI

loader = GoogleClassroomLoader(
    course_ids=["123456789"],
    load_attachments=True,
    vision_model=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
)
```

### Student Submissions, Topics, and Roster

```python
loader = GoogleClassroomLoader(
    course_ids=["123456789"],
    load_submissions=True,
    load_topics=True,
    load_roster=True,
)
```

## Features

- **Full Classroom API coverage** — assignments, announcements, materials,
  submissions, rubrics, topics, and roster
- **Drive attachments** — PDF, DOCX, CSV, text, and image parsing with
  Google Docs/Slides/Sheets export
- **Vision LLM** — embedded images described by Gemini, GPT-4V, or any
  vision-capable `BaseChatModel`
- **YouTube and link attachments** — metadata captured as structured documents
- **Pluggable parsers** — bring your own `BaseBlobParser` (PyMuPDF, Unstructured, etc.)
- **File size guard** — configurable `max_file_size` to skip oversized attachments
- **Retry with backoff** — exponential backoff with jitter on HTTP 429/500/503
- **Flexible auth** — service accounts, OAuth, cached tokens, or pre-built credentials
- **Rich metadata** — course info, timestamps, due dates, grades, links on every document
- **Lazy and async loading** — `lazy_load()` and `alazy_load()` for memory efficiency
- **Pydantic v2** — fully typed `BaseModel` with `model_dump()`, JSON schema, and
  automatic scope injection

For full documentation and API reference, see the
[GitHub repository](https://github.com/ayanokojix21/langchain-google-classroom).
