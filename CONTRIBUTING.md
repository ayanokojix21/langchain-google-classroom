# Contributing to langchain-google-classroom

Thank you for your interest in contributing. This package follows LangChain
integration conventions and aims to provide a robust Google Classroom document
loader for RAG pipelines and educational AI.

For broader ecosystem guidance, see the
[LangChain contributing overview](https://docs.langchain.com/oss/python/contributing/overview).

## Development Setup

Requires **Python >=3.10**.

```bash
git clone https://github.com/ayanokojix21/langchain-google-classroom.git
cd langchain-google-classroom/libs/google-classroom

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -e ".[dev]"
```

## Running Tests

```bash
# Unit tests
pytest tests/unit_tests/ -v --disable-socket --allow-unix-socket

# Unit tests with coverage
pytest tests/unit_tests/ --cov=langchain_google_classroom --cov-report=term-missing

# Integration tests (requires credentials)
pytest tests/integration_tests/ -m integration -v
```

## Linting and Formatting

```bash
ruff format .
ruff check .
ruff check . --fix
```

All code must pass `ruff check` and `ruff format --check` before merging.

## Code Style

- Type annotations on all functions and methods.
- Docstrings in Google style on all public APIs.
- `from __future__ import annotations` at the top of every module.
- `guard_import()` for optional dependencies (pypdf, python-docx).
- `BaseBlobParser` + `Blob` interface for all file parsers.
- `BaseLoader.lazy_load()` as the primary entry point.

## Adding a New Parser

1. Create `langchain_google_classroom/parsers/your_parser.py`.
2. Implement `BaseBlobParser.lazy_parse(blob)` → `Iterator[Document]`.
3. Add the MIME type mapping in `parsers/__init__.py`.
4. Add tests in `tests/unit_tests/test_parsers.py`.
5. Run `pytest` and `ruff check`.

## Pull Request Process

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Make changes with tests.
4. Run the full test suite and linter.
5. Commit with a conventional message (`feat:`, `fix:`, `docs:`).
6. Open a pull request. All CI checks must pass before merging.

## Reporting Issues

Use [GitHub Issues](https://github.com/ayanokojix21/langchain-google-classroom/issues):

- **Bug reports** — include environment, steps to reproduce, expected vs actual
  behaviour, and traceback.
- **Feature requests** — describe the use case and proposed API.