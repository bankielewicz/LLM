# S0 — Contracts and custody

S0 supplies the reproducible foundation for the intermediate implementation. It preserves all 107 pre-existing files, validates the sealed contracts and fixtures, compiles complete curriculum editions from authored source, and binds dependency installations to four concrete wheel inventories.

The application implementation continues in S1 through S5. S0 does not provide the companion service, the authored applied lessons, native model execution, or native browser acceptance. The 605 product acceptance execution units remain NOT_RUN.

## Authority

- Approved revision 1.1: commit acb04d173ab1c4c6395ac95c95ad04b8fa50e849; payload f611dddea0addf4da6329b2157a8adbb0458614d92c59b9341610921f4918e62.
- Working revision 1.2: the same intermediate-v1 edition, with the four-profile summary and explicit role-specific error-status bindings reconciled under the owner's 2026-10-01 instruction to proceed.
- The exact manifest and payload identities are in [authority.json](authority.json). The unmodified revision 1.1 manifest is retained beside it; the custody gate recomputes all 150 original normative file hashes from the immutable Git commit.
- The current 107-file source inventory stays byte-exact for this slice. S0 does not relax the source gate or modify existing frontend files.
- Scope and amendment records are retained under the specification's reviews directory. Product approval and qualification do not follow from a successful S0 gate.

The scoped implementation review and corrected findings are recorded in [review.md](review.md).

## Run the gate

Use Python 3.12 with the pinned authoring dependencies from companion/locks/dev.requirements.txt. The gate checks the running environment against every package version in that lock. The separate development environment does not install Torch or download a model. See [dependencies.md](dependencies.md) for exact installation and lock verification commands.

From the repository root, run:

    python -B scripts/run_s0_gate.py \
      --report-dir /absolute/path/to/a/new/attempt-directory \
      --node /absolute/path/to/node \
      --canonical-course /absolute/path/to/llm-foundations-v2 \
      --legacy-python /absolute/path/to/legacy-test-environment/bin/python

The report directory must not already exist. Every attempt retains stdout, stderr, a source manifest, gate statuses, and a result.json. A source change during execution fails the attempt. Missing setup, test skips, an empty unit-test denominator, a failed check, or a blocked mandatory gate cannot produce PASS.

On the current Windows/WSL host, the existing Windows Node executable can run the unchanged JavaScript regression suite from WSL using:

    --node "/mnt/c/Program Files/nodejs/node.exe"

The canonical course is available at /home/bryan/Projects/llm-foundations-v2. The gate reads its receipt-bound files; it never modifies that package.

## What is checked

1. Approved specification custody, active specification custody, and all 107 protected source hashes, before and after execution.
2. Requirement/case index consistency, closed schema and OpenAPI references, fixtures, semantic rules, error bindings, edge cases, independent fixture arithmetic, materialized fixture bytes, and the specification seal.
3. The complete S0 Python suite, including compiler failures, dependency/wheel mismatches, and negative custody controls.
4. The unchanged 36 JavaScript foundation/import regressions.
5. All seven original Python lab regressions in a separate Python 3.12 / torch 2.11.0+cpu environment.
6. The unchanged source accessibility and course/ZIP audit.

The original lab tests exercise tiny CPU training, exact resume, evaluation, and generation. These are legacy compatibility controls, not companion jobs or intermediate product qualification. The separate legacy environment uses the original course's Torch pin; it does not change the companion's D10 Torch 2.8.0 requirements.

The historical source audit writes qa/accessibility-source.json, which is a frozen file. S0 runs the byte-identical auditor and templates in a disposable copy, retains their output, and leaves the worktree's protected bytes untouched. No browser is launched by that audit.

Generated specification reports have fresh names below reviews/. They are excluded by the seal, retained as attempts, and copied into the gate's result directory.

## Components

- scripts/check_custody.py: anchored source and specification preservation.
- scripts/compile_applied.py: deterministic edition compiler; [compiler contract](compiler.md).
- companion/src/llm_foundations_companion/contracts.py: offline contract validation; [validation scope](contracts.md).
- companion/locks/: per-profile install locks, concrete wheel manifests, and their validator; [dependency custody](dependencies.md).
- companion/tests/: source and failure-path tests.
- scripts/run_s0_gate.py: one retained gate across the complete S0 source candidate.

The compiler has valid and malformed temporary 24-module test editions. S0 deliberately ships no placeholder lessons or generated applied-content.json; the later curriculum slices supply the real source material.

## Continuing in S1

Use the frozen operation schemas, error bindings, limits, and source gate as inputs. Implement the authenticated loopback service, storage root, sessions, dataset registry, one-worker jobs, cancellation, errors/events, and crash recovery in the prescribed slice order.

The review-harness dependency and GPU measurements remain historical evidence. B-04 fit calibration on all four reference profiles and a product-produced device-comparison-reference-v1 snapshot, sealed through a later revision, remain mandatory before qualification. E-01 through E-06 in the review backlog are separate future expansions.

Any implementation change creates a new candidate. Re-run affected checks and retain failed attempts; a previous PASS does not qualify changed bytes.
