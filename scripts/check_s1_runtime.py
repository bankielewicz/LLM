"""Record and verify an isolated locked companion runtime environment."""
from __future__ import annotations
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

from llm_foundations_companion import __version__ as companion_version


def normalize(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def main():
    manifest_path = Path(sys.argv[1])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {normalize(row['name']): row['version'] for row in manifest['packages']}
    # The release package owns its version. Keeping this check tied to the
    # installed package avoids teaching later slices a historical S1 version
    # while still comparing the installed distribution metadata below.
    expected['llm-foundations-companion'] = companion_version
    installed = {}
    for distribution in metadata.distributions():
        name = normalize(distribution.metadata['Name'])
        if name in installed:
            raise RuntimeError('Duplicate installed distribution: ' + name)
        installed[name] = distribution.version
    failures = [name for name, version in expected.items() if installed.get(name) != version]
    failures.extend(name for name in installed if name not in expected and name != 'pip')
    if sys.version_info[:2] != (3, 12) or platform.python_implementation() != 'CPython':
        failures.append('python')
    identity = {'python': sys.version, 'executable': sys.executable,
        'executable_sha256': hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        'lock_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'packages': [{'name': name, 'version': version} for name, version in sorted(installed.items())],
        'mismatches': sorted(failures)}
    print(json.dumps(identity, sort_keys=True))
    if failures:
        return 1
    return subprocess.run([sys.executable, '-I', '-m', 'pip', 'check'], check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
