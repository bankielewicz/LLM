"""Generate/check packaged profile identities from the existing frozen locks."""
import argparse
import hashlib
import json
from pathlib import Path


def expected(root):
    profiles = {}
    for profile in ("win-cpu", "win-cuda", "wsl-cpu", "wsl-cuda"):
        lock = (root / "companion" / "locks" / f"{profile}.requirements.txt").read_bytes()
        wheels = (root / "companion" / "locks" / f"{profile}.wheels.json").read_bytes()
        digest = hashlib.sha256(lock).hexdigest()
        assert json.loads(wheels)["lock_sha256"] == digest, profile
        profiles[profile] = {"dependency_lock_sha256": digest, "wheel_manifest_sha256": hashlib.sha256(wheels).hexdigest()}
    return (json.dumps({"format": "llmf-runtime-lock-manifest-v1", "profiles": profiles}, sort_keys=True, indent=2) + "\n").encode()


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "companion/src/llm_foundations_companion/runtime_data/runtime-lock-manifest.json"
    raw = expected(root)
    if args.write:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(raw)
    assert output.read_bytes() == raw, "Packaged runtime lock identities differ from the frozen locks"
    print("runtime identities: PASS (4 exact requirements digests and 4 wheel manifests)")


if __name__ == "__main__":
    main()
