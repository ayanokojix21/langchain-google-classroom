# Contributing to langchain-google-classroom

Thank you for your interest in contributing! This guide will help you get started.

We welcome contributions from the community.
This package follows LangChain integration conventions and aims to provide a robust Google Classroom loader for RAG pipelines.
Before submitting a PR, please review the guidelines below.

For broader ecosystem guidance, see the LangChain contributing overview:
https://docs.langchain.com/oss/python/contributing/overview

## Development Setup

Requires **Python >=3.10**.

```bash
# Clone the repository
git clone https://github.com/ayanokojix21/langchain-google-classroom.git
cd langchain-google-classroom

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -m integration -v

# Run a specific test file
pytest tests/unit/test_parsers.py -v

# Run with coverage
pytest tests/unit/ --cov=langchain_google_classroom --cov-report=term-missing
```

Note: integration tests may require local credentials such as service account or OAuth files.

## Linting and Formatting

```bash
# Format code
ruff format .

# Check for lint errors
ruff check .

# Auto-fix issues
ruff check . --fix
```

## Code Style

This project follows LangChain coding conventions:

- **Type annotations** on all functions and methods
- **Docstrings** in Google/NumPy style on all public functions
- **`from __future__ import annotations`** at the top of every module
- **`guard_import`** for optional dependencies (pypdf, python-docx)
- **`BaseBlobParser` + `Blob`** interface for all file parsers
- **`BaseLoader.lazy_load()`** as the main entry point

## Adding a New Parser

1. Create `langchain_google_classroom/parsers/your_parser.py`
2. Implement `BaseBlobParser.lazy_parse(blob)` → `Iterator[Document]`
3. Add the MIME type mapping in `parsers/__init__.py`
4. Add tests in `tests/unit/test_parsers.py`
5. Run `pytest` and `ruff check`

## Tests

All pull requests should include tests for new functionality or bug fixes.

- Add or update unit tests for behavior changes.
- Add integration tests when external API behavior changes.
- Keep tests deterministic and avoid flaky assertions.

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes with tests
4. Run `pytest tests/ -v` and `ruff check`
5. Ensure CI checks pass before requesting merge
6. Commit with a descriptive message
7. Push and open a Pull Request

All pull requests must pass CI checks before merging.

Suggested commit message style:

- `feat: add markdown parser`
- `fix: handle empty attachments`
- `docs: improve README`

## Reporting Issues

Use [GitHub Issues](https://github.com/ayanokojix21/langchain-google-classroom/issues) with:

- **Bug reports**: Include environment, steps to reproduce, expected behavior, actual behavior, and traceback
- **Feature requests**: Describe the use case and proposed API