# S2 tiny-model operations

This work slice adds six tiny-model operations under the sealed revision 1.2
contracts and the separately owner-approved
[APP-009 revision 1.3 amendment](../../specs/intermediate-v1-amendments/APP-009-revision-1.3.md).
It starts from the S1 merge bound by [`baseline.json`](baseline.json). The
composite authority in [`authority.json`](authority.json) preserves the sealed
specification, baseline records, original proposal, and protected foundation
files byte for byte.

## Delivered surface

An installed companion wheel can advertise and run exactly six operations:
`tokenizer_train`, `tiny_train`, `tiny_resume`, tiny-subject `evaluate`,
tiny-backend `context_preview`, and `generate`.
The existing authenticated API and owner `request` command submit all six.
They remain distinct operations with separately bound inputs, runs, artifacts,
and terminal results.

Tokenizer training reads the admitted training split and registers an immutable
tokenizer artifact and identity. Tiny training binds the selected dataset
splits, tokenizer, fixed recipe, runtime profile, dependency lock, and installed
companion source revision before queue insertion. The service process stays
framework-free; model and optimizer work occurs only in the fixed owned worker.

Tiny checkpoints contain exactly the seven specified JSON and safetensors
files. Parent admission checks their membership, sizes, digests, and JSON
identities. The fixed worker checks tensor names, shapes, dtypes, optimizer
state, random-number state, and permitted tied aliases before effective state
use. Checkpoint publication and worker
acknowledgement follow the durable registry commit. Resume creates a new child
run, preserves the parent checkpoint identity, restores model, optimizer,
schedule, and random-number state, and continues from the recorded step. It
does not rewrite the parent run.

Tiny evaluation reads the exact admitted validation or test split, scores every
record and target including EOS under the byte-normalized profile, and does not
update weights. The parent verifies the summary metrics and the complete
subject-major record artifact against that split before publishing either.

Tiny `context_preview` selects a prospective `checkpoint_id` and tokenizes
the prompt without loading model weights. `generate` verifies the matching
preview and checkpoint, performs deterministic greedy or seeded top-p decoding,
and preserves weights. A stale preview is rejected at admission or rechecked
by the fixed worker before use. The resolved subject identity still contains
its separate `model_id`; the amendment changes only the client request field.
Applied-model, chat, retrieval, and bundle operations remain unavailable.

S3 supplies the learner interface, forms, navigation, and evidence flow; this
slice does not add browser training controls. The E01 learner-source verifier,
its receipt registration, and receipt-gated tied training profile are deferred
to S5. S2 does not execute or accept learner Python source.

## Run the installed slice

Use a Python 3.12 environment prepared from the matching complete runtime lock
in `companion/locks/`. Build a wheel only from a captured source candidate, then
install that wheel with `--no-deps` into the prepared runtime environment. Do
not install it into the global Python environment.

The service verifies the loaded package against the installed distribution,
its exact wheel `RECORD`, embedded build provenance, source revision, package
digest, and selected profile lock. A source-checkout launch has no installed
build identity and advertises the six S2 operations as unavailable. Source
tests are development evidence; they are not an installed execution path.

The current command surface is:

```text
python -m llm_foundations_companion gpu-check
python -m llm_foundations_companion serve --storage <new-storage-root> --profile <win-cpu|win-cuda|wsl-cpu|wsl-cuda>
python -m llm_foundations_companion connect --storage <storage-root>
python -m llm_foundations_companion status --storage <storage-root>
python -m llm_foundations_companion request --storage <storage-root> --file <request.json>
python -m llm_foundations_companion request --storage <storage-root> --file <request.json> --idempotency-key <uuid>
python -m llm_foundations_companion cancel --storage <storage-root> --job-id <uuid>
python -m llm_foundations_companion events --storage <storage-root> --job-id <uuid> --after <decimal-cursor>
```

`--profile` is optional; when supplied it must be one of the four values shown.
`request.json` must be one closed `JobRequest` from the wheel-local OpenAPI
contract. Omitting `--idempotency-key` creates a new UUID, while an explicit
UUID identifies a deliberate retry. `events --after` requires an ASCII decimal
cursor. `serve` retains the S1 fixed loopback origin, private owner control
channel, root lease, pairing behavior, and storage-recovery rules.

## Validation and scope

The implementation review is recorded in [`review.md`](review.md). Its focused
source tests cover the inspected operation, checkpoint, scheduler, worker, and
service boundaries. They do not qualify an installed wheel or a native runtime
profile.

S2 retains its implemented publication gate of at least 95% line coverage
across the affected executable component's complete source tree, with
reproducible commands and no exclusion of untested production logic.
Coverage is an additional publication gate; the source and native subsets do
not replace it. For this slice the measured component is the complete
`companion/src/llm_foundations_companion/` Python tree, including bundled lab
sources. The measurement combines source tests, installed service and worker
traces, and the unchanged legacy suite with its child processes. Instrumented
runtimes are disposable copies; ordinary locked-runtime execution remains a
separate check.

Pass the sealed measurement receipt to `scripts/run_s2_gate.py` with
`--coverage-result <coverage-attempt>/result.json`, together with the gate's
runtime, legacy-runtime, Node, Windows-Python, canonical-course, and report
arguments. The gate checks the separately approved amendment before and after execution,
exact source, commit, tree, wheel and evidence identities, the complete
Python-file inventory, zero excluded lines, and an unrounded coverage ratio
of at least 95 percent. All 25 native cases must pass; prior evidence that
retains seven APP-009-blocked cases cannot satisfy the amended gate. A missing receipt is blocked;
a stale, incomplete, or failing receipt fails the coverage check. Retain the
measurement scripts, commands, configuration, phase receipts, raw traces, and
hash indexes with that receipt so the measurement can be reproduced.

The gate's raw-trace verifier requires Coverage.py in its authoring process.
Use the separately prepared development-tooling copy containing every pinned
development dependency plus the receipt-bound coverage tool. The coverage
campaign currently uses Coverage.py 7.13.1. Clear coverage tracing variables
when running the final gate. Keep `--runtime-python` pointed at the ordinary
locked companion runtime and `--legacy-python` at the separate locked legacy
runtime. This extra authoring tool does not change either runtime lock or
establish native qualification for an instrumented environment.

Installed results are candidate-specific. The final bound gate report retains
the source manifest, wheel, interpreter, dependency lock, process output, and
result links outside the source worktree; the S2 pull request links that report.
This README intentionally carries no candidate result count that could be
mistaken for a later wheel.

The frozen product acceptance registry still contains **605 execution units,
all `NOT_RUN` at this source-review boundary**. The four native profiles
(`win-cpu`, `win-cuda`, `wsl-cpu`, and `wsl-cuda`) likewise remain `NOT_RUN`
here. A candidate-specific subset report does not reduce that denominator or
establish product qualification, browser acceptance, release readiness, or
owner acceptance.
