# S0 dependency declarations and locks

## Scope

This slice declares the companion package and freezes dependency inputs. It does
not install any D10 runtime profile, start the companion, run a model, or
qualify a product candidate. S0 may create the separate legacy test environment
to execute the protected original course regressions.

The runtime declaration follows D10 exactly. The project dependencies include
`torch==2.8.0`, which admits the selected PEP 440 local build, and the seven
other exact pins. Each profile extra further constrains Torch to `+cpu` or
`+cu128`. Contract validation and test tooling are confined to the `dev`
extra and the separate dev lock.

## Runtime lock structure

Each release profile has two files under `companion/locks/`:

- `<profile>.requirements.txt` is the complete transitive, hash-required
  resolver lock copied byte-for-byte from the approved review gate.
- `<profile>.wheels.json` selects exactly one compatible CPython 3.12 wheel for
  every locked package and binds its filename, source index, URL, SHA-256 and
  byte count. It also binds the requirements lock by SHA-256 and records the
  total wheel download bytes.

The second file is required because a pip multi-hash requirements file permits
several compatible artifacts and therefore does not, by itself, bind one wheel
filename. The manifest generator reads package metadata and HTTP headers only;
it does not download wheel bodies. The committed manifest is the frozen input.
Regenerating it is an explicit lock update that creates new candidate bytes.

The future setup builder must download only the manifest URLs, verify filename,
length and SHA-256, and then install from that verified wheelhouse with network
access disabled:

```text
python -m pip install --no-index --find-links <verified-wheelhouse> \
  --require-hashes -r companion/locks/<profile>.requirements.txt
```

Using a verified wheelhouse makes the manifest's one-artifact selection
effective while pip independently enforces the sealed multi-hash dependency
lock. CPU Torch is sourced from `https://download.pytorch.org/whl/cpu`; CUDA
Torch and its Linux CUDA dependencies are sourced from
`https://download.pytorch.org/whl/cu128`. Other packages retain the index
selected by the sealed resolver output.

`companion/locks/materialize_wheelhouse.py` performs the exact-URL download
into a new destination and verifies every byte before publishing a wheel
filename. A failed attempt keeps received `.part` bytes and writes
`failed-download.json` with expected and actual size and SHA-256 evidence.
`validate.py --wheelhouse` provides a separate offline readback.

A bare `pip install .` or `pip install .[<profile>]` is not a supported setup
path because it would resolve transitive artifacts from ambient indexes. Setup
must use the profile manifest and lock. The S1 startup preflight remains
responsible for rejecting an environment whose selected profile, operating
system or installed Torch build does not match.

## Development dependencies

`dev.in` holds the exact direct authoring, test and no-isolation build pins.
`dev.requirements.txt` is its complete transitive hash lock. These packages are
absent from the project's runtime dependency declaration unless a D10 amendment
adds them. In particular, `jsonschema` and `referencing` support explicit
contract checks; importing the companion package must not require them.

## Protected legacy regression environment

The original course independently declares `torch==2.11.0`. Its
`legacy-test.in`, hash lock and wheel manifest select
`torch==2.11.0+cpu` for CPython 3.12 on WSL. This environment is used only for
`course/labs/test_lab.py`; it is not a companion release profile and does not
alter D10's Torch 2.8.0 selection.

## Provenance and verification

The four runtime requirements files come from the read-only review package
`validation/gates/lock-<profile>.txt`. Their package counts are 39 for
win-cpu, 39 for win-cuda, 38 for wsl-cpu and 53 for wsl-cuda, matching the
sealed dependency evidence in the approved specification.

Run the complete offline check from the repository root:

```text
python companion/locks/validate.py
```

Exact dev, legacy test and verified-wheelhouse commands are in
`companion/locks/README.md`.

The check covers direct pins, profile counts, sealed lock digests, selected
wheel compatibility, source indexes, artifact hashes and byte totals. The
pytest file also proves that hashless, multi-index and unknown active lock lines
are rejected by both lock consumers.

These checks establish static lock consistency. The specification's native
install, package-consistency, startup/import, fit-calibration and full acceptance
obligations remain separate qualification work.
