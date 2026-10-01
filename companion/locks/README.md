# Dependency lock usage

All commands run from the repository root with Python 3.12. The four companion
runtime profiles use their committed requirements lock and wheel manifest.
`dev` and `legacy-test` are separate environments and are not release
profiles.

## Offline declaration check

```bash
python3.12 companion/locks/validate.py
```

This command performs no network access and does not import Torch.

## Development and contract-check environment

```bash
python3.12 -m venv envs/dev
envs/dev/bin/python -m pip install \
  --require-hashes \
  --only-binary=:all: \
  -r companion/locks/dev.requirements.txt
envs/dev/bin/python -m pytest -q companion/tests
envs/dev/bin/python -m build --no-isolation --wheel companion
```

The dev lock includes the exact build backend versions used by
`companion/pyproject.toml`. It contains no Torch or companion runtime
dependency.

## Protected legacy course regression environment

The legacy course declares Torch 2.11.0 independently of D10. Materialize its
exact CPython 3.12 WSL CPU wheel set from the committed URLs into a destination
that does not already exist:

```bash
python3.12 -m venv envs/legacy-test
python3.12 companion/locks/materialize_wheelhouse.py \
  legacy-test .wheelhouse/legacy-test
python3.12 companion/locks/validate.py \
  --wheelhouse legacy-test .wheelhouse/legacy-test
```

Install only after the wheelhouse check passes:

```bash
envs/legacy-test/bin/python -m pip install \
  --no-index \
  --find-links .wheelhouse/legacy-test \
  --require-hashes \
  -r companion/locks/legacy-test.requirements.txt
envs/legacy-test/bin/python -m unittest discover \
  -s course/labs -p 'test_lab.py'
```

The wheelhouse validator requires exactly the ten manifest filenames and checks
the size and SHA-256 of every file. The manifest selects
`torch-2.11.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl`.

## Companion runtime profiles

Do not use bare `pip install .` or `pip install .[<profile>]`. The release
builder must construct a new wheelhouse from the exact committed URLs:

```bash
python3.12 companion/locks/materialize_wheelhouse.py \
  <profile> <verified-wheelhouse>
```

If a transfer or verification fails, the new destination is retained as an
attempt. Received bytes remain under `<filename>.part`, and
`failed-download.json` records the actual and expected byte counts and SHA-256,
the artifact identity and the failure reason. Retry with another new
destination; only a verified partial file is promoted to its wheel filename.

It then passes that directory to:

```bash
python3.12 companion/locks/validate.py \
  --wheelhouse <profile> <verified-wheelhouse>
```

and only then install with the selected environment's interpreter. For WSL:

```bash
envs/<profile>/bin/python -m pip install \
  --no-index \
  --find-links <verified-wheelhouse> \
  --require-hashes \
  -r companion/locks/<profile>.requirements.txt
```

For native Windows, the equivalent command is:

```powershell
.\envs\<profile>\Scripts\python.exe -m pip install `
  --no-index `
  --find-links <verified-wheelhouse> `
  --require-hashes `
  -r companion/locks/<profile>.requirements.txt
```

S0 freezes and checks these inputs. It does not install any of the four D10
runtime profiles.
