"""Embed immutable source provenance in built distributions, never the checkout."""
import hashlib
import json
from pathlib import Path
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent

def source_digest():
    root = ROOT / "src" / "llm_foundations_companion"
    rows = [{"path": p.relative_to(root).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size_bytes": p.stat().st_size}
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    rows.sort(key=lambda row: row["path"])
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def provenance():
    retained = ROOT / "build-provenance.json"
    if retained.is_file():
        value = json.loads(retained.read_text(encoding="utf-8"))
        if value["package_source_sha256"] != source_digest():
            raise RuntimeError("Source archive differs from its retained build provenance")
        return value
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    return {"format": "llmf-build-provenance-v1", "source_revision": git("rev-parse", "HEAD"),
            "source_tree": git("rev-parse", "HEAD^{tree}"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
            "package_source_sha256": source_digest()}

def write_provenance(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")

class BoundBuild(build_py):
    def run(self):
        value = provenance()
        super().run()
        write_provenance(Path(self.build_lib) / "llm_foundations_companion" / "runtime_data" / "build-provenance.json", value)

class BoundSdist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        write_provenance(Path(base_dir) / "build-provenance.json", provenance())

setup(cmdclass={"build_py": BoundBuild, "sdist": BoundSdist})
