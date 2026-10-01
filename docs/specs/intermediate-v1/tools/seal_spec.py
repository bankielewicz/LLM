"""Seal or verify the normative specification payload, never product behavior."""
from pathlib import Path
import argparse, hashlib, json, sys
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "spec-manifest.json"
def current():
    rows = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        parts = path.relative_to(ROOT).parts
        if rel == "spec-manifest.json" or parts[0] == "reviews" or "__pycache__" in parts[:-1] or path.suffix == ".pyc":
            continue
        raw = path.read_bytes()
        rows.append({"path": rel, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    rows.sort(key=lambda x: x["path"])
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {"format": "llm-foundations-spec-manifest-v1",
            "spec_revision": json.loads((ROOT / "requirements.json").read_text(encoding="utf-8"))["spec_revision"],
            "base_commit": "3a47ea48da53cb9de3ff4727ca5f0f0b6f2b9bf8",
            "exclusions": ["reviews/**", "spec-manifest.json", "**/__pycache__/**", "**/*.pyc"],
            "digest_rule": "SHA256 of UTF-8 json.dumps(files,sort_keys=True,separators=(comma,colon),ensure_ascii=False), no final LF; files sorted by relative POSIX path",
            "files": rows, "payload_sha256": hashlib.sha256(canonical).hexdigest()}
ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true")
args = ap.parse_args()
data = current()
if args.check:
    expected = json.loads(TARGET.read_text(encoding="utf-8"))
    if expected != data:
        print("FAIL: normative payload differs from sealed manifest")
        sys.exit(1)
else:
    with TARGET.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, indent=2, ensure_ascii=False)+"\n")
print(json.dumps({"result":"PASS", "files":len(data["files"]), "payload_sha256":data["payload_sha256"]}))
