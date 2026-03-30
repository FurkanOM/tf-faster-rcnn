"""Run the repository's local lint and unit-test commands."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def run_command(command: Sequence[str]) -> int:
    """Execute a command from the repository root and stream its output.

    Args:
        command (Sequence[str]): Command and arguments to execute.

    Returns:
        int: Process exit code.
    """
    print("$ {}".format(" ".join(command)))
    completed = subprocess.run(command, cwd=ROOT)
    return completed.returncode


def main() -> int:
    """Run linting first, then unit tests, and return the failing status if any.

    Returns:
        int: Process exit code.
    """
    commands = [
        [sys.executable, "tools/lint.py"],
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
    ]

    for command in commands:
        return_code = run_command(command)
        if return_code != 0:
            return return_code

    return 0


if __name__ == "__main__":
    sys.exit(main())
