"""Bind every packaged companion file to the candidate's source bytes."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def main():
    wheel, repo = Path(sys.argv[1]), Path(sys.argv[2])
    source = repo / 'companion/src/llm_foundations_companion'
    expected = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        assert len(members) == len(set(members)), 'Duplicate wheel member'
        actual = {name.removeprefix('llm_foundations_companion/'): archive.read(name)
                  for name in members if name.startswith('llm_foundations_companion/') and not name.endswith('/')}
    assert expected == actual, 'Wheel/source member or byte mismatch: ' + str(sorted(expected.keys() ^ actual.keys()))
    print(json.dumps({'package_files': len(expected), 'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(), 'source_bytes_match': True}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
