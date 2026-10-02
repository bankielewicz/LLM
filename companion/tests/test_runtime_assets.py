from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "dist"
AUTHORED_SOURCE = REPOSITORY / "scripts" / "runtime_assets"
PACKAGE = (
    REPOSITORY
    / "companion"
    / "src"
    / "llm_foundations_companion"
    / "static"
)
BUILDER = REPOSITORY / "scripts" / "build_runtime_assets.py"


def _run_builder(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BUILDER), *(str(item) for item in arguments)],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )


def _strict_manifest() -> dict[str, object]:
    def reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate manifest key: {key}")
            result[key] = value
        return result

    return json.loads(
        (PACKAGE / "manifest.json").read_text(encoding="utf-8", errors="strict"),
        object_pairs_hook=reject_pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            AssertionError(f"non-finite manifest number: {token}")
        ),
    )


def test_runtime_asset_builder_is_byte_exact() -> None:
    completed = _run_builder("--check")
    assert completed.returncode == 0, completed.stderr
    assert "49 published + 3 authored assets" in completed.stdout


def test_asset_manifest_is_complete_public_inventory() -> None:
    manifest = _strict_manifest()
    assert manifest["format"] == "llm-foundations-runtime-assets-v1"
    rows = manifest["files"]
    authored_rows = manifest["authored_files"]
    assert isinstance(rows, list)
    assert isinstance(authored_rows, list)
    names = [row["path"] for row in rows]
    authored_names = [row["path"] for row in authored_rows]
    assert names == sorted(names)
    assert authored_names == sorted(authored_names)
    assert len(names) == len(set(names)) == 49
    assert len(authored_names) == len(set(authored_names)) == 3
    assert set(names).isdisjoint(authored_names)

    source_names = sorted(
        path.relative_to(SOURCE).as_posix()
        for path in SOURCE.rglob("*")
        if path.is_file()
    )
    package_names = sorted(
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*")
        if path.is_file() and path != PACKAGE / "manifest.json"
    )
    source_authored_names = sorted(
        path.relative_to(AUTHORED_SOURCE).as_posix()
        for path in AUTHORED_SOURCE.rglob("*")
        if path.is_file()
    )
    assert names == source_names
    assert authored_names == source_authored_names
    assert package_names == sorted(names + authored_names)
    assert {"index.html", "app.js", "core.js", "styles.css", "course.zip"} <= set(names)
    assert any(name.startswith("course/lessons/") for name in names)
    assert any(name.startswith("course/labs/") for name in names)

    forbidden_parts = {"spec", "specs", "scratch", "private", "reviews", "__pycache__"}
    for row in [*rows, *authored_rows]:
        name = row["path"]
        assert forbidden_parts.isdisjoint(part.casefold() for part in Path(name).parts)
        assert not name.casefold().endswith((".pyc", ".pyo"))

    for row in rows:
        name = row["path"]
        source = SOURCE / name
        packaged = PACKAGE / name
        raw = source.read_bytes()
        assert packaged.read_bytes() == raw
        assert row == {
            "path": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "source": f"dist/{name}",
        }

    for row in authored_rows:
        name = row["path"]
        source = AUTHORED_SOURCE / name
        packaged = PACKAGE / name
        raw = source.read_bytes()
        assert packaged.read_bytes() == raw
        assert row == {
            "path": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "source": f"scripts/runtime_assets/{name}",
        }


def test_builder_rejects_an_unreviewed_source_file(tmp_path: Path) -> None:
    source = tmp_path / "dist"
    output = tmp_path / "static"
    shutil.copytree(SOURCE, source)
    (source / "private-notes.txt").write_text("must not be bundled", encoding="utf-8")
    completed = _run_builder("--source", source, "--output", output, "--write")
    assert completed.returncode == 1
    assert "unreviewed published asset private-notes.txt" in completed.stderr
    assert not output.exists()


def test_builder_rejects_an_unreviewed_authored_file(tmp_path: Path) -> None:
    authored_source = tmp_path / "runtime_assets"
    output = tmp_path / "static"
    shutil.copytree(AUTHORED_SOURCE, authored_source)
    (authored_source / "debug.js").write_text("secret = true", encoding="utf-8")
    completed = _run_builder(
        "--authored-source", authored_source, "--output", output, "--write"
    )
    assert completed.returncode == 1
    assert "unreviewed authored asset debug.js" in completed.stderr
    assert not output.exists()


def test_local_shell_preserves_reader_and_contains_credentials() -> None:
    legacy_html = (SOURCE / "index.html").read_text(encoding="utf-8")
    local_html = (AUTHORED_SOURCE / "local-index.html").read_text(encoding="utf-8")
    session_js = (AUTHORED_SOURCE / "session.js").read_text(encoding="utf-8")

    assert (PACKAGE / "index.html").read_text(encoding="utf-8") == legacy_html
    assert '<script type="module" src="app.js"></script>' in legacy_html
    assert 'src="app.js"' not in local_html
    assert '<script type="module" src="session.js"></script>' in local_html
    assert '<link rel="stylesheet" href="styles.css">' in local_html
    assert '<link rel="stylesheet" href="session.css">' in local_html
    for preserved in ('class="skip"', 'id="app"', 'id="toast"'):
        assert preserved in local_html

    assert session_js.index("history.replaceState") < session_js.index("fetch(")
    assert session_js.index("history.replaceState") < session_js.index('import("./app.js")')
    assert "window.sessionStorage" in session_js
    assert "localStorage" not in session_js
    assert "indexedDB" not in session_js
    assert "statusText.textContent = text" in session_js
    assert "innerHTML" not in session_js
    assert "console" not in session_js
    assert 'Authorization: "Bearer " + session.access_token' in session_js
    assert '"X-LLMF-CSRF": current.csrf_token' in session_js
    assert "clearStoredSessions();" in session_js
    for label in (
        "Pair this tab",
        "Checking local runtime",
        "Local runtime connected",
        "Connected · storage read-only",
        "Incompatible local runtime",
        "Session expired · pair again",
        "Connection lost",
    ):
        assert label in session_js or label in local_html


def _detached_project_with_provenance(tmp_path: Path) -> tuple[Path, dict]:
    project = tmp_path / "companion"
    shutil.copytree(
        REPOSITORY / "companion",
        project,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "build", "*.egg-info"),
    )
    package = project / "src" / "llm_foundations_companion"
    rows = [{"path": path.relative_to(package).as_posix(),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "size_bytes": path.stat().st_size}
            for path in sorted(package.rglob("*")) if path.is_file()]
    rows.sort(key=lambda row: row["path"])
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(REPOSITORY), *args], text=True).strip()
    provenance = {
        "format": "llmf-build-provenance-v1",
        "source_revision": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
        "package_source_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    (project / "build-provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return project, provenance


def test_wheel_contains_complete_reader_without_cache_files(tmp_path: Path) -> None:
    project, provenance = _detached_project_with_provenance(tmp_path)
    wheelhouse = tmp_path / "wheelhouse"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--outdir",
            str(wheelhouse),
            str(project),
        ],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = list(wheelhouse.glob("*.whl"))
    assert len(wheels) == 1

    prefix = "llm_foundations_companion/static/"
    with zipfile.ZipFile(wheels[0]) as archive:
        members = set(archive.namelist())
        assert json.loads(archive.read("llm_foundations_companion/runtime_data/build-provenance.json")) == provenance
        packaged = {name.removeprefix(prefix) for name in members if name.startswith(prefix)}
    expected = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*")
        if path.is_file()
    }
    assert packaged == expected
    assert not any("__pycache__" in name or name.endswith((".pyc", ".pyo")) for name in members)


def test_detached_wheel_rejects_changed_source_after_provenance_capture(tmp_path: Path) -> None:
    project, _ = _detached_project_with_provenance(tmp_path)
    module = project / "src" / "llm_foundations_companion" / "__init__.py"
    module.write_bytes(module.read_bytes() + b"\n# modified after source capture\n")
    completed = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--wheel", "--outdir", str(tmp_path / "wheelhouse"), str(project)],
        cwd=REPOSITORY, check=False, capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "Source archive differs from its retained build provenance" in completed.stdout + completed.stderr
