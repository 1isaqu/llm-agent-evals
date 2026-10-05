"""Properties of the repository itself: no secrets, no data files, one place for the key."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "llm_agent_evals"
FORBIDDEN_SUFFIXES = {
    ".csv",
    ".tsv",
    ".parquet",
    ".pkl",
    ".joblib",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".xlsx",
    ".zip",
    ".png",
    ".jpg",
    ".pdf",
}


def tracked_files():
    """Files git would add: tracked or untracked and not ignored. Falls back to a walk."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.splitlines()
        return [ROOT / line for line in out if line]
    except (OSError, subprocess.SubprocessError):
        skip = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
        return [p for p in ROOT.rglob("*") if p.is_file() and not skip & set(p.parts)]


FILES = [p for p in tracked_files() if p.exists()]


def test_no_data_files_or_binaries_are_part_of_the_repository():
    bad = [p.relative_to(ROOT).as_posix() for p in FILES if p.suffix.lower() in FORBIDDEN_SUFFIXES]
    assert bad == []


def test_the_env_example_holds_placeholders_only():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "LLM_API_KEY=\n" in text
    assert "DB_PASSWORD=change-me" in text and "DB_HOST=localhost" in text
    assert not re.search(
        r"(sk-[A-Za-z0-9]{12,}|AKIA[0-9A-Z]{12,}|Bearer\s+[A-Za-z0-9._-]{20,})", text
    )


def test_the_gitignore_covers_secrets_data_and_environments():
    patterns = set((ROOT / ".gitignore").read_text(encoding="utf-8").split())
    for required in (
        ".env",
        ".venv/",
        "node_modules/",
        "data/",
        "*.csv",
        "*.parquet",
        "*.pkl",
        "*.joblib",
        "mlruns/",
        "dist/",
    ):
        assert required in patterns, required


def test_there_is_no_env_file_in_the_tree():
    assert not [p for p in FILES if p.name == ".env"]


def test_the_api_key_variable_is_read_in_exactly_one_module():
    readers = [
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*.py")
        if re.search(r"""["']LLM_API_KEY["']""", p.read_text(encoding="utf-8"))
    ]
    assert readers == ["settings.py"]


def test_no_source_file_contains_a_credential_shaped_literal():
    pattern = re.compile(
        r"(sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
    )
    for path in FILES:
        if path.suffix in {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".html", ".example"}:
            assert not pattern.search(path.read_text(encoding="utf-8")), path


def test_runtime_dependencies_are_empty_and_the_database_driver_is_optional():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "dependencies = []" in text
    assert re.search(r'sqlserver = \["pyodbc', text)


def test_only_the_sql_server_helper_imports_the_database_driver():
    importers = [
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*.py")
        if re.search(r"^\s*import pyodbc", p.read_text(encoding="utf-8"), re.MULTILINE)
    ]
    assert importers == ["sources/sqlsource.py"]


def test_source_files_are_ascii_so_no_stray_encoding_surprises_exist():
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert text.isascii(), path


def test_every_package_directory_has_an_init_file():
    for directory in [p for p in SRC.rglob("*") if p.is_dir() and p.name != "__pycache__"]:
        assert (directory / "__init__.py").exists(), directory


@pytest.mark.parametrize(
    "name",
    [
        "README.md",
        "LICENSE",
        ".gitignore",
        ".env.example",
        ".github/workflows/ci.yml",
        "docs/architecture.md",
        "docs/data-contract.md",
        "docs/glossary.md",
        "docs/decisions.md",
        "docs/eval-format.md",
    ],
)
def test_required_files_exist(name):
    assert (ROOT / name).is_file()


def test_the_license_is_mit_with_the_expected_holder():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License") and "Copyright (c) 2026 Isaque Carvalho" in text
