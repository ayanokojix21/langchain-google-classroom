"""Check which library directories have changed in a PR/push.

Used by CI workflows to determine which packages need testing.
Based on the LangChain integration-repo-template pattern.
"""

from __future__ import annotations

import subprocess
import sys

# Library directories to monitor for changes.
LIB_DIRS = ["libs/google-classroom"]


def _get_changed_files(base_ref: str = "origin/main") -> list[str]:
    """Get list of files changed compared to base_ref."""
    result = subprocess.run(
        ["git", "diff", "--name-only", base_ref],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().split("\n")


def main() -> None:
    base_ref = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    changed_files = _get_changed_files(base_ref)

    changed_libs: list[str] = []
    for lib_dir in LIB_DIRS:
        if any(f.startswith(lib_dir) for f in changed_files):
            changed_libs.append(lib_dir)

    if changed_libs:
        print(",".join(changed_libs))  # noqa: T201
    else:
        print("none")  # noqa: T201


if __name__ == "__main__":
    main()
