"""Project-local lint checks that run without third-party tooling."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
from typing import Iterator, List


ROOT = Path(__file__).resolve().parents[1]
CHECK_DIRS = ("models", "tests", "tools", "utils")
CHECK_FILES = (
    "__init__.py",
    "faster_rcnn_predictor.py",
    "faster_rcnn_trainer.py",
    "rpn_predictor.py",
    "rpn_trainer.py",
)


def iter_python_files() -> Iterator[Path]:
    """Yield tracked Python files that should be covered by local checks.

    Returns:
        Iterator[Path]: Python files included in the local quality checks.
    """
    files = {ROOT / filename for filename in CHECK_FILES}
    for directory in CHECK_DIRS:
        files.update((ROOT / directory).glob("*.py"))
    for path in sorted(files):
        if path.exists():
            yield path


def check_syntax(path: Path, source: str) -> List[str]:
    """Validate that a Python file can be compiled successfully.

    Args:
        path (Path): File path being checked.
        source (str): Source code contents.

    Returns:
        List[str]: Collected lint issues for syntax failures.
    """
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        return [f"{path.relative_to(ROOT)}:{exc.lineno}: syntax error: {exc.msg}"]
    return []


def check_trailing_whitespace(path: Path, source: str) -> List[str]:
    """Reject tabs and trailing whitespace that make reviews noisier.

    Args:
        path (Path): File path being checked.
        source (str): Source code contents.

    Returns:
        List[str]: Collected lint issues for whitespace problems.
    """
    issues = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        stripped_line = line.rstrip("\n")
        if stripped_line != stripped_line.rstrip():
            issues.append(f"{path.relative_to(ROOT)}:{line_number}: trailing whitespace")
        if "\t" in line:
            issues.append(f"{path.relative_to(ROOT)}:{line_number}: tab character found")
    return issues


def check_docstrings(path: Path, tree: ast.Module) -> List[str]:
    """Require module and public top-level definitions to be documented.

    Args:
        path (Path): File path being checked.
        tree (ast.Module): Parsed abstract syntax tree.

    Returns:
        List[str]: Collected lint issues for missing docstrings.
    """
    issues = []
    relative_path = path.relative_to(ROOT)
    if ast.get_docstring(tree) is None:
        issues.append(f"{relative_path}:1: missing module docstring")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            if ast.get_docstring(node) is None:
                issues.append(
                    f"{relative_path}:{node.lineno}: missing docstring for `{node.name}`"
                )
    return issues


def check_annotations(path: Path, tree: ast.Module) -> List[str]:
    """Require function arguments and return values to be annotated.

    Args:
        path (Path): File path being checked.
        tree (ast.Module): Parsed abstract syntax tree.

    Returns:
        List[str]: Collected lint issues for missing annotations.
    """
    issues = []
    relative_path = path.relative_to(ROOT)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
        for argument in arguments:
            if argument.arg in {"self", "cls"}:
                continue
            if argument.annotation is None:
                issues.append(
                    f"{relative_path}:{node.lineno}: missing type annotation for `{node.name}` argument `{argument.arg}`"
                )
        if node.args.vararg is not None and node.args.vararg.annotation is None:
            issues.append(
                f"{relative_path}:{node.lineno}: missing type annotation for `{node.name}` vararg `*{node.args.vararg.arg}`"
            )
        if node.args.kwarg is not None and node.args.kwarg.annotation is None:
            issues.append(
                f"{relative_path}:{node.lineno}: missing type annotation for `{node.name}` kwarg `**{node.args.kwarg.arg}`"
            )
        if node.returns is None:
            issues.append(f"{relative_path}:{node.lineno}: missing return type annotation for `{node.name}`")
    return issues


def main() -> int:
    """Run all lint checks and print findings in a grep-friendly format.

    Returns:
        int: Process exit code.
    """
    issues = []
    for path in iter_python_files():
        source = path.read_text(encoding="utf-8")
        issues.extend(check_syntax(path, source))
        issues.extend(check_trailing_whitespace(path, source))
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        issues.extend(check_docstrings(path, tree))
        issues.extend(check_annotations(path, tree))

    if issues:
        print("\n".join(issues))
        return 1

    print("lint: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
