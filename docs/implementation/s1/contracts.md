# S1 runtime contracts and bundled reader

Status: implemented and covered by focused development tests in the pinned WSL
Python 3.12 environment. Product acceptance remains `NOT_RUN`.

## Sealed runtime contracts

[`build_runtime_contracts.py`](../../../scripts/build_runtime_contracts.py)
copies exactly 36 JSON metadata documents selected from the immutable
revision-1.2 package into `llm_foundations_companion.contract_data`. It checks
each source byte count and SHA-256 against `spec-manifest.json` before writing.
The generated manifest binds every wheel-local filename to its source path,
size, digest, spec revision, and sealed payload digest. Runtime reads verify the
manifest declaration, byte count, and digest, and return defensive copies.
The additional `materialized-manifest.json` is the authoritative shipped-fixture
path/digest/record-count registry used by storage provenance checks. Its JSONL
fixture text is not copied into the contract bundle.

The stdlib-only runtime validator rejects malformed UTF-8, duplicate JSON
members, `NaN`, infinities, and finite-looking numeric tokens that overflow to
infinity. Canonical JSON uses the Python 3.12 contract: sorted keys, compact
separators, UTF-8, no trailing line feed, and no Unicode normalization. The
schema evaluator resolves only bundled documents, rejects unsupported
validation keywords, enforces the frozen Draft 2020-12 keyword profile,
normalizes integers to floats only where a schema requires `number`, and runs
the pure semantic rules required before service/storage eligibility checks.
Relative document references are POSIX-normalized against their owning bundled
document. Absolute paths, URI schemes, query strings, backslashes, NULs, and
names that escape the bundle root are rejected; resolution never retrieves an
external document.
Timestamps follow the normative runtime rule: UTC `Z` with exactly three
millisecond digits.

`ApiError` resolves the closed error vocabulary by role. Top-level HTTP codes
and nested reason codes retain distinct status/retry bindings, so a code such as
`PAYLOAD_TOO_LARGE` cannot be flattened across its transport and semantic uses.

## Public reader assets

[`build_runtime_assets.py`](../../../scripts/build_runtime_assets.py) carries an
explicit allowlist of the 49 already-published `dist/**` files. It copies those
bytes into `llm_foundations_companion.static`. It also copies three authored
local-shell files from `scripts/runtime_assets`: `local-index.html`,
`session.css`, and `session.js`. The builder generates a deterministic
compact JSON inventory that keeps published and authored rows separate, with
source paths, byte counts, and SHA-256 values. The
allowlist includes the published legacy course tree and `course.zip`, so the
reader's existing download links keep their exact public bytes. The source
`dist/**` tree is never rewritten.

The local shell removes a `#/connect/{secret}` fragment before its first API
fetch or dynamic reader import, exchanges it for instance-scoped credentials,
and retains those credentials only in `sessionStorage`. Its connection banner
uses the frozen learner-facing labels and DOM text assignment. Expiry,
revocation, instance mismatch, and explicit disconnect clear stored session
records. The original `index.html`, `app.js`, `core.js`, styles, content,
and course downloads remain byte-identical to `dist/**`.

The builder fails when the source has a missing, symbolic-link, or unreviewed
file; its check mode also rejects missing, changed, symbolic-link, or unexpected
package files. This closed inventory prevents repository specs, review records,
scratch files, caches, and private material from entering the wheel through a
recursive source glob. The wheel configuration names the legacy
`course/.gitignore` explicitly because setuptools does not include that public
hidden file through its recursive glob.

## Development verification

Run the focused checks from the repository root with the pinned S0 development
environment:

```text
/home/bryan/Projects/LLM-worktrees/s0-validation/preparation-20261001/dev-env/bin/python -m pytest companion/tests/test_runtime_contracts.py companion/tests/test_runtime_assets.py -q
/home/bryan/Projects/LLM-worktrees/s0-validation/preparation-20261001/dev-env/bin/python scripts/build_runtime_contracts.py --check
/home/bryan/Projects/LLM-worktrees/s0-validation/preparation-20261001/dev-env/bin/python scripts/build_runtime_assets.py --check
```

The focused contract suite covers all 49 frozen schema-fixture expectations,
the exact runtime limits, defensive loads, strict/canonical JSON, the complete
57-operation OpenAPI parameter/request/response schema inventory, representative
Draft 2020-12 oracle comparisons, all 683 references across the complete
bundled inventory resolved from their owning documents, cross-document OpenAPI joins, semantic byte
limits, type normalization, and role-aware errors. The asset suite checks byte
identity and hashes, rejects
unreviewed published and authored source files, checks the shell's
fragment/session boundaries, builds a real wheel, verifies every static member,
and rejects cache bytecode in that wheel.

These checks establish source-to-package contract custody, static validator
behavior, and wheel membership for the S1 candidate. They do not establish
browser rendering, live reader/companion pairing, accessibility, installation
on a target profile, model execution, or any of the 605 product acceptance
execution units. The authored shell therefore remains a statically verified S1
candidate; native-browser behavior is still `NOT_RUN`.
