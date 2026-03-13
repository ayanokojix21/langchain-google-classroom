# langchain-google-classroom

An integration package connecting **Google Classroom** and **LangChain**.

Load Google Classroom content — assignments, announcements, course materials, and
attachments — as LangChain `Document` objects for use in RAG pipelines, semantic
search, AI teaching assistants, and course chatbots.

## Installation

```bash
pip install langchain-google-classroom
```

For file attachment parsing (PDF, DOCX):

```bash
pip install langchain-google-classroom[parsers]
```

## Quickstart

```python
from langchain_google_classroom import GoogleClassroomLoader

# Load all accessible courses
loader = GoogleClassroomLoader()
docs = loader.load()

# Load specific courses
loader = GoogleClassroomLoader(course_ids=["123456789"])
docs = loader.load()

# Selective loading
loader = GoogleClassroomLoader(
    course_ids=["123456789"],
    load_assignments=True,
    load_announcements=True,
    load_materials=False,
)
docs = loader.load()
```

## Authentication

The loader supports multiple authentication methods:

### Service Account (recommended for production)

```python
loader = GoogleClassroomLoader(
    service_account_file="path/to/service_account.json",
)
```

### OAuth User Credentials

```python
loader = GoogleClassroomLoader(
    client_secrets_file="path/to/credentials.json",
    token_file="path/to/token.json",
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

## Document Structure

Each document contains:

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
        "due_date": "2024-01-22T23:59:00Z",
        "alternate_link": "https://classroom.google.com/...",
    }
)
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `course_ids` | `List[str]` | `None` | Specific course IDs (None = all) |
| `load_assignments` | `bool` | `True` | Load courseWork items |
| `load_announcements` | `bool` | `True` | Load announcements |
| `load_materials` | `bool` | `True` | Load courseWorkMaterials |
| `credentials` | `Credentials` | `None` | Pre-built Google credentials |
| `service_account_file` | `str` | `None` | Path to service account JSON |
| `token_file` | `str` | `None` | Path to cached OAuth token |
| `client_secrets_file` | `str` | `None` | Path to OAuth client secrets |
| `scopes` | `List[str]` | Classroom read scopes | API scopes |

## License

MIT