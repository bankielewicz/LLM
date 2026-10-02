"""Bind every packaged companion file and generated build identity to its source."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    wheel, repo = args.wheel, args.repo
    source = repo / "companion/src/llm_foundations_companion"
    expected = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        assert len(members) == len(set(members)), "Duplicate wheel member"
        actual = {name.removeprefix("llm_foundations_companion/"): archive.read(name)
                  for name in members if name.startswith("llm_foundations_companion/") and not name.endswith("/")}
    provenance = actual.pop("runtime_data/build-provenance.json", None)
    assert expected == actual, "Wheel/source member or byte mismatch: " + str(sorted(expected.keys() ^ actual.keys()))
    report = {"package_files": len(expected), "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "source_bytes_match": True}
    if provenance is not None:
        value = json.loads(provenance)
        assert set(value) == {"format", "source_revision", "source_tree", "dirty", "package_source_sha256"}
        assert value["format"] == "llmf-build-provenance-v1" and type(value["dirty"]) is bool
        def git(*argv):
            return subprocess.check_output(["git", "-C", str(repo), *argv], text=True).strip()
        rows = [{"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)} for name, raw in sorted(expected.items())]
        assert value["package_source_sha256"] == hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert value["source_revision"] == git("rev-parse", "HEAD")
        assert value["source_tree"] == git("rev-parse", "HEAD^{tree}")
        assert value["dirty"] == bool(git("status", "--porcelain", "--untracked-files=all"))
        if args.require_clean:
            assert not value["dirty"], "Final installed candidate must have a clean committed source tree"
        report["build_provenance"] = value
    elif args.require_clean:
        raise AssertionError("Final S2 wheel lacks generated source provenance")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
