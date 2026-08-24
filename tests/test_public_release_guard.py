from __future__ import annotations

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOTS = (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "tests",
    ROOT / "docs",
    ROOT / ".github",
    ROOT / "data" / "public",
)
PUBLIC_DATA_ROOT = ROOT / "data" / "public"
PUBLIC_DATA_FILES = {
    "shandong_market_2024_public.xlsx",
    "shandong_market_2025_public.xlsx",
    "shandong_market_2026-01-01_2026-08-15_public.xlsx",
    "manual_realtime_prices_2026-08-13_2026-08-22_public.xlsx",
}
PUBLIC_FILES = (
    ROOT / "README.md",
    ROOT / "README.zh-CN.md",
    ROOT / "AGENTS.md",
    ROOT / "pyproject.toml",
    ROOT / "LICENSE",
    ROOT / "NOTICE",
    ROOT / "THIRD_PARTY_NOTICES.md",
    ROOT / "CITATION.cff",
)
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".cff", ".example", ".txt", ".json"}
FORBIDDEN_PATTERNS = (
    re.compile(r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:[\\/]|\\\\[^\\/\s]+[\\/])"),
    re.compile(
        r"(?i)(?:AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|"
        r"github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|"
        r"xox[baprs]-[A-Za-z0-9-]{20,}|"
        r"-----BEGIN (?:RSA|EC|OPENSSH|PRIVATE) KEY-----)"
    ),
)
FORBIDDEN_SUFFIXES = {".parquet", ".xlsx", ".xls", ".xlsm", ".pt", ".pth", ".ckpt", ".onnx", ".npz"}
IGNORED_SCAN_PARTS = {".git", ".venv", ".pytest_cache", ".private-runtime", "dist", "build"}
SELF_GUARD_FILES = {
    Path(__file__).resolve(),
    ROOT / ".github" / "workflows" / "ci.yml",
    ROOT / "scripts" / "ingest_public_shandong_workbooks.py",
}


def _prospective_public_files() -> set[Path]:
    """Scan tracked files plus the public roots before they are committed."""
    result = set(PUBLIC_FILES)
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result.update(ROOT / value for value in tracked.stdout.splitlines())
    for root in PUBLIC_ROOTS:
        result.update(path for path in root.rglob("*") if path.is_file())
    return result


def test_public_text_does_not_contain_private_machine_markers() -> None:
    for path in _prospective_public_files():
        if path.name == "uv.lock" or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.resolve() in SELF_GUARD_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(pattern.search(text) for pattern in FORBIDDEN_PATTERNS), path


def test_public_tree_has_no_private_artifact_suffixes() -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(
            part in IGNORED_SCAN_PARTS
            or part.startswith(".tmp-")
            or part.startswith(".pytest-tmp")
            or part.startswith(".uv-cache")
            or part.startswith(".venv-public-")
            for part in path.parts
        ):
            continue
        if path.suffix.lower() == ".xlsx":
            assert path.parent.resolve() == PUBLIC_DATA_ROOT.resolve(), path
            assert path.name in PUBLIC_DATA_FILES, path
            continue
        assert path.suffix.lower() not in FORBIDDEN_SUFFIXES, path


def test_public_data_package_has_exact_allowlisted_workbooks() -> None:
    assert PUBLIC_DATA_ROOT.is_dir()
    assert {path.name for path in PUBLIC_DATA_ROOT.glob("*.xlsx")} == PUBLIC_DATA_FILES
    assert (PUBLIC_DATA_ROOT / "MANIFEST.json").is_file()


def test_public_data_workbooks_are_not_ignored() -> None:
    for name in PUBLIC_DATA_FILES:
        path = PUBLIC_DATA_ROOT / name
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", str(path.relative_to(ROOT))],
            cwd=ROOT,
        )
        assert result.returncode != 0, path


def test_private_runtime_is_explicitly_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".private-runtime/" in gitignore
