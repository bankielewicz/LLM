# S2 implementation review

This record describes the scoped source review before the retained,
candidate-specific S2 gate. It is not installed-native qualification, product
qualification, or owner acceptance.

This is the retained pre-amendment review. References below to the four enabled
operations and the APP-009 hold describe that historical candidate. The owner
has since approved the exact proposal; the separate revision 1.3 amendment and
`authority.json` record the effective change. The amended six-operation
candidate requires fresh source, installed-native, coverage, and final-gate
evidence; historical counts below do not qualify it.

## Review scope and disposition

- **Capability and provenance boundary:** independent review confirmed that the
  capability table enables only `tokenizer_train`, `tiny_train`, `tiny_resume`,
  and tiny-subject `evaluate`. APP-009-held preview and generation remain
  unavailable before job creation. S2 admission requires a matching installed
  wheel identity; a source checkout has no companion source revision and keeps
  the four operations unavailable.
- **Admission and immutable inputs:** review traced closed request validation,
  reference resolution, dataset eligibility, tokenizer and checkpoint
  identities, profile and lock binding, immutable input materialization, and
  byte/row reservations before queue insertion. The service process does not
  import a model framework, and the worker receives only parent-verified fixed
  inputs.
- **Checkpoint and resume safety:** review inspected the seven-file checkpoint
  writer, parent-side admission, and fixed-worker loading. The parent checks
  JSON metadata, exact file membership, digests, sizes, step, and lineage. The
  worker validates safetensors structure, model and optimizer tensors,
  random-number state, and allowed aliases before state use. Resume restores
  the exact accepted state into a new child run.
  The worker advances its incumbent only after the parent durably commits and
  acknowledges the checkpoint.
- **Durable effects and rollback:** review traced staged artifacts through
  checkpoint and terminal transactions. Verified staged bytes survive a
  database rollback for exact retry, while pending custody and staging are
  cleared only after commit. Run results use the durable job step, checkpoint
  boundary, and finish time rather than accepting contradictory worker claims.
  Completed training must reach its requested final step.
- **Evaluation:** review confirmed whole-split tiny evaluation, EOS-inclusive
  byte normalization, read-only model use, typed finite metrics, and exact
  subject-major evaluation records. Parent validation binds record bytes,
  cardinality, order, arithmetic, metric totals, subject count, and the records
  artifact identity to the admitted split before paired publication.
- **Failure and limit behavior:** review covered cancellation checkpoints,
  process loss, failed terminal outputs, monotonic metric sequence and step,
  storage reservations, logical row limits, and free-space preflight. Invalid
  or incomplete terminal effects do not acquire successful registry identity.

Confirmed defects at these joins were repaired and rechecked. In particular,
checkpoint manifests are now validated during synchronous resume admission;
staged terminal and checkpoint bytes remain retryable across transaction
rollback; cleanup runs only after commit; terminal run records are derived from
durable job state; completed training cannot stop short of its requested step;
and evaluation record publication is checked against the exact admitted split
and metric artifact.

## Exact source-test evidence

The following bounded tests were run against their recorded development
snapshots; the final gate separately binds the eventual committed candidate:

- Retained development unit attempt `unit-development-attempt-001`: **398
  passed, 1 native-Windows-only case skipped, 25 subtests passed, 0 failures,
  and 0 errors in 18.77 seconds**. Its before/after inventory contained **485
  identical source members** with manifest digest
  `5e2144e3b670e49452e5f2d27b1ffbb52f0a5fbd1d88904ed760367f26e89cf5`.
- `test_tiny_evaluation_records.py` plus `test_operation_store.py`: **21
  passed**. This covered the evaluation-record validator and its parent commit
  integration, including strict field types and artifact binding.
- `test_tiny_evaluation_records.py`, `test_operation_store.py`,
  `test_training_store.py`, `test_scheduler.py`, `test_worker_protocol.py`, and
  `test_service.py`: **84 passed in 10.71 seconds**. This exercised the joined
  storage, admission, scheduling, worker protocol, and service paths after the
  reviewed fixes.

The full suite and focused subsets establish only their stated source-test
scopes. They used development test environments and do not show that a built wheel ran
the operations through a real installed service.

## Subsequent development validation

Later development attempts exercised the installed service and fixed workers,
then added explicit failure-boundary tests. These attempts remain separate from
the final committed-candidate gate. Failed and incomplete attempts are retained
outside the source worktree; later success does not replace their observations.

Installed execution exposed two integration defects that the earlier source
subsets did not detect. Training metrics now use the sealed long-form metric
schema, with typed name, value, unit, protocol, and timestamp fields. The
scheduler now persists at most sixteen progress milestones while continuing to
update the job's observed step for every valid worker progress message. The
forty-step cadence check verifies all 202 metric rows, 41 metric events, and 41
checkpoint events. Invalid terminal artifacts now fail the owned job through
the worker protocol failure path, preserving previously acknowledged
checkpoints and allowing the next queued job to dispatch.

Additional malformed-input tests checked preflight enum types, checkpoint
metadata and tensors, upload/archive boundaries, event-stream revocation and
disconnection, contract-bundle integrity, and public error-envelope bindings.
The preflight parser now rejects non-string enum values through its bounded
validation path. The internal generation handler translates malformed preview
JSON to `CONTEXT_PREVIEW_STALE`; its public capability remains disabled under
the APP-009 hold.

On retained development wheel attempt 008, the eighteen unblocked native cases
and all sixteen supporting tests passed. The other seven of the fixed
twenty-five native cases remained explicitly blocked by APP-009. The
uninterrupted fifty-step and resumed twenty-five-plus-twenty-five runs produced
identical model, optimizer, and RNG files and the same final validation loss.
The sixteen installed S1 service regression checks also passed on that wheel.
These results are historical development evidence: the subsequent malformed
preview repair requires a new wheel and fresh candidate validation.

Independent coverage review found that the first combined unit/native report
measured 48 importable modules but omitted six bundled lab Python files. Its
86.45 percent result is a subset result, not the required complete-tree result.
The corrected workflow includes all 54 bound Python files, keeps exclusions
empty, instruments a disposable copy of the legacy runtime, and verifies raw
service and worker execution plus exact source-path binding. At this development-review boundary, the coverage gate
remains pending until that complete inventory is measured on the final source
candidate. The later candidate-specific gate result is the authority for its
final disposition.

A later retained instrumented attempt exposed a job-read failure when the host
wall clock moved backward by 537 milliseconds between starting and running.
The resulting update time preceded the job creation time, so the public
response correctly failed its contract check. That failed attempt remains
retained; subsequent completion of the same job does not turn it into a pass.

The scheduler repair floors each existing job mutation against its persisted
SQL and job-document timestamps. The same chosen time binds the job update,
event, and transition; terminal preparation and publication reuse that time
for job and run-result metadata. Recovery obtains the floor from persisted
state. Deterministic regressions reproduce the original rollback and cover
terminal consistency, cancellation, and restart with newer SQL timestamps than
the job document. Queue selection retains the approved ordering by
`created_at` then `job_id`. The rebuilt candidate requires fresh native and
coverage evidence; earlier wheel results do not qualify this repair.

## Qualification boundary

No installed-native PASS is claimed by this source review. Installed execution
must use a wheel built from the captured candidate and a matching locked runtime
environment. Its retained gate report, stored outside the source worktree and
linked from the pull request, is the authority for candidate identity, exact
test denominator, native observations, failures, and `NOT_RUN` cases.

All **605** product acceptance execution units and all four native profiles
remain `NOT_RUN` at this source-review boundary. Source tests cannot promote
them. S3 learner-interface behavior, S5 E01 source verification and tied-profile
workflow, complete browser acceptance, fit calibration, release distribution,
and owner acceptance remain separate obligations.

The four delivered operations do not make S2 complete while APP-009 remains
unauthorized and tiny context preview and generation remain unavailable. No
sealed specification, implementation plan, baseline receipt, or product
acceptance status was changed by this review.
