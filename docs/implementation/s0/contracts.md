# S0 contract validation

Status: implemented as an offline source and test boundary. This check does not start the local companion or establish product qualification.

The validator reads the authoritative revisioned package at `docs/specs/intermediate-v1` in place. Contract files are not copied into the Python package in S0. Service integration and runtime request validation remain S1 work.

Run it from the repository root with the S0 development dependencies installed:

```text
PYTHONPATH=companion/src python -m llm_foundations_companion.contracts --spec-root docs/specs/intermediate-v1
```

The command emits one JSON object and exits nonzero when any check fails. `--pretty` produces indented output for review. The implementation is [contracts.py](../../../companion/src/llm_foundations_companion/contracts.py), with focused tests in [test_contracts.py](../../../companion/tests/test_contracts.py).

## Closed validation boundary

The validator performs these checks without a network retrieval callback:

- strict UTF-8 JSON parsing, rejecting duplicate object keys, `NaN`, infinities, and finite-looking numbers that overflow to infinity;
- Draft 2020-12 compilation of the external schemas, model profile, OpenAPI component schemas, and inline operation schemas;
- resolution of every `$ref` against the owning local document, rejecting network, absolute, query-bearing, escaping, missing, and unresolved references;
- a bounded OpenAPI 3.1.0 profile: the fixed loopback server, path and operation structure, unique operation IDs, responses, and compiled schemas;
- every declared schema fixture against its registered schema, including the expected valid or invalid result;
- the pinned SmolLM2 model-profile fixture against `model-profile.json`, its identity fields against the download manifest, and `download_manifest_sha256` against the exact manifest bytes;
- retained materialized-fixture byte counts, SHA-256 values, record counts, strict JSONL rows, and the mechanically bound semantic fixture oracles;
- OpenAPI `x-semantic-rules` bindings against the closed semantic-rule registry;
- error-vocabulary role shapes, OpenAPI enum equality, reason-to-top-level status bindings, and every declared semantic rejection.

The error vocabulary is resolved by `(code, kind)`. In particular, `PAYLOAD_TOO_LARGE` is a top-level transport boundary at HTTP 413 and a semantic reason under `VALIDATION_FAILED` at HTTP 400. A single flattened lookup is invalid for a multi-kind code.

## Current source result

The command reports 35 contract JSON documents, 577 schema units, 683 references, 57 operations, 49 of 49 declared data-schema fixtures plus 1 of 1 pinned model-profile fixture matching their contracts, 3 semantic fixture cases with 8 mechanically bound checks, 14 materialized files, 33 semantic rules with 25 OpenAPI annotations, and 104 error codes. Its source-hash inventory includes both the pinned profile instance and the exact model download manifest.

The negative controls prove rejection of duplicate and non-finite JSON; missing and network references; invalid `~2` escapes; leading-zero, non-ASCII, and out-of-bounds array indices; an invalid artifact fixture; a malformed model profile; a stale download-manifest digest; a fixture whose expected outcome contradicts its schema result; an unknown semantic-rule annotation; and a collapsed 413 semantic `PAYLOAD_TOO_LARGE` binding.

These results establish S0 contract consistency for the checked source bytes. They do not execute an API request, storage transaction, browser behavior, model operation, dependency installation, or any of the 605 product acceptance execution units.
