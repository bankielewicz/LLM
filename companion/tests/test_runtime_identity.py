"""Installed-source identity rejects source shadows and package drift."""
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_foundations_companion import runtime_identity


def installed_fixture(tmp_path, monkeypatch):
    root = tmp_path / "llm_foundations_companion"
    (root / "runtime_data").mkdir(parents=True)
    (root / "runtime_identity.py").write_text("# installed fixture\n")
    rows = [{"path": "runtime_identity.py", "sha256": hashlib.sha256((root / "runtime_identity.py").read_bytes()).hexdigest(), "size_bytes": (root / "runtime_identity.py").stat().st_size}]
    value = {"format": "llmf-build-provenance-v1", "source_revision": "a" * 40, "source_tree": "b" * 40, "dirty": False, "package_source_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    (root / "runtime_data/build-provenance.json").write_text(json.dumps(value))
    class Entry:
        def __init__(self, path):
            self.name = path.relative_to(tmp_path).as_posix()
            self.hash = SimpleNamespace(mode="sha256", value=base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b"=").decode())
            self.size = path.stat().st_size
        def __str__(self):
            return self.name
    dist = SimpleNamespace(files=[Entry(path) for path in root.rglob("*") if path.is_file()], locate_file=lambda name: tmp_path / name, read_text=lambda name: "Wheel-Version: 1.0" if name == "WHEEL" else None)
    monkeypatch.setattr(runtime_identity, "__file__", str(root / "runtime_identity.py"))
    monkeypatch.setattr(runtime_identity.metadata, "distribution", lambda name: dist)
    monkeypatch.setattr(runtime_identity, "_read", lambda name: json.loads((root / "runtime_data" / name).read_text()))
    return root, value, dist


def test_loaded_package_requires_matching_installed_wheel(tmp_path, monkeypatch):
    root, value, dist = installed_fixture(tmp_path, monkeypatch)
    assert runtime_identity.build_provenance() == value
    dist.locate_file = lambda name: tmp_path / "different-install" / name
    with pytest.raises(ValueError, match="not the installed wheel"):
        runtime_identity.build_provenance()


@pytest.mark.parametrize("mutation", ["changed", "extra", "missing", "link"])
def test_installed_package_drift_is_rejected(tmp_path, monkeypatch, mutation):
    root, value, dist = installed_fixture(tmp_path, monkeypatch)
    if mutation == "changed":
        (root / "runtime_identity.py").write_text("# drift\n")
    elif mutation == "extra":
        (root / "unreviewed.py").write_text("# extra\n")
    elif mutation == "missing":
        (root / "runtime_identity.py").unlink()
    else:
        (root / "linked.py").symlink_to(root / "runtime_identity.py")
    with pytest.raises(ValueError):
        runtime_identity.build_provenance()


def test_source_stamp_without_installed_distribution_is_rejected(tmp_path, monkeypatch):
    installed_fixture(tmp_path, monkeypatch)
    def missing(name):
        raise runtime_identity.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(runtime_identity.metadata, "distribution", missing)
    with pytest.raises(ValueError, match="requires an installed"):
        runtime_identity.build_provenance()
