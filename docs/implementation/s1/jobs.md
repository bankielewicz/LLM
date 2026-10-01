# S1 jobs, events, and fixed worker boundary

Status: implemented source slice. This note describes the S1 implementation and
its bounded verification. It does not claim the product-acceptance cases in the
sealed specification have run.

## Admission and atomic acceptance

`Scheduler.submit` validates the closed OpenAPI request component, canonicalizes
the validated request, and checks the instance/method/resolved-route idempotency
record before capability admission. An exact retained 2xx response is returned
byte-for-byte before a capability check. A changed request digest under that
scope returns `IDEMPOTENCY_CONFLICT`. Records expire after 24 hours, survive
session expiry, and never store a 4xx or 5xx response.

For a newly accepted job, one SQLite transaction:

1. rechecks idempotency and the eight-queued-job bound;
2. reserves byte and registry-row capacity;
3. writes and registers the immutable canonical request artifact;
4. inserts the queued job and cursor-1 `state_changed` event;
5. records the exact 202 response; and
6. advances the registry revision.

Rollback therefore exposes none of those rows as an accepted job. The registry
owns object bytes, capacity accounting, artifact integrity, and the authenticated
cursor codec. The scheduler owns `jobs`, `job_events`, and
`idempotency_records`. Public job records are serialized in `jobs.record_json`;
indexed columns mirror the fixed filter and order fields.

All 15 production operation handlers deliberately raise
`CAPABILITY_UNAVAILABLE` in S1. The production capability map reports the same
state, so a new job is rejected before artifact, reservation, job, event, or
idempotency mutation. Later operation slices must replace only fixed handler
imports and capability records; request data cannot select a module or callable.

## FIFO state and recovery

The scheduler permits one `starting`, `running`, or `cancelling` worker and up
to eight queued jobs. FIFO order is `(created_at ASC, job_id ASC)`. Job-list
order is `(created_at DESC, job_id DESC)`, with an HMAC cursor bound to the
instance, schema, order, and complete filter object.

Queued cancellation writes `cancelling`, `interrupted`, and `terminal` events in
one transaction, starts no worker, and releases its reservation. Running
cancellation keeps the first reason, opens the 30-second cooperative grace, and
then terminates only the retained owned container. A 60-minute deadline begins
at `starting`. Recovery runs before sessions, control IPC, or scheduling;
`starting`, `running`, and `cancelling` rows become `interrupted` with
`service_restarted`, while queued jobs remain FIFO. A job/event cursor mismatch
puts scheduling into read-only recovery rather than rewriting evidence.
If the cancellation pipe is already broken, the scheduler resolves the retained
container and records `worker_lost` instead of letting the pipe error escape. If
storage becomes read-only while a worker is active, service shutdown performs no
metadata write: it terminates the retained container and clears the active
ownership record only after termination succeeds. A failed termination retains
that ownership record so cleanup can be retried before the root lease closes.

## Fixed process and inherited channels

Production argv is exactly:

```text
<qualified interpreter> -I -m llm_foundations_companion.worker_main --job-id <uuid> --instance-id <uuid>
```

The working directory is the installed package root. The environment is rebuilt
from an allowlist containing UTF-8 settings, fixed runtime controls, the selected
profile, job/instance process identities, staging path, and inherited channel
identities. Proxy, credential, Python-path, shell-startup, and arbitrary parent
variables are absent. Canonical request data crosses a one-MiB framed request
pipe. Protocol output and cancellation use separate inherited framed pipes;
stdin is closed and stdout/stderr are each captured to a one-MiB prefix plus
total-byte/truncation metadata.

On WSL/Linux, `subprocess` creates the process group without `preexec_fn`. At
worker start, before reading request data, the child installs `PDEATHSIG=SIGKILL`
and verifies the allowlisted parent PID. On Windows, only the converted pipe and
lease handles appear in `STARTUPINFO.lpAttributeList.handle_list`. The parent
creates a kill-on-close Job Object and assigns the child before sending request
bytes. The retained process group or Job Object, plus a 256-bit spawn nonce, is
the only termination authority. A missing or mismatched proof pauses scheduling
and records `worker_ownership_unknown`; there is no PID/name scan.

The inherited instance-lease descriptor/handle remains open until the worker is
gone, preventing another service instance from taking the root while a child can
still write.

## Protocol and events

Every request envelope binds protocol version, job ID, instance ID, spawn nonce,
schema ID, canonical request SHA-256, and the closed operation request. The
worker verifies all identities and the digest before `ready`. The parent accepts
exactly one matching `ready`, then typed events and one operation-matching
result, job error, or interruption. Data before `ready`, a second acknowledgment,
output after a terminal message, a nonfinite metric, incoherent checkpoint pair,
unknown state/phase/reason, or operation mismatch is
`WORKER_PROTOCOL_ERROR`.

A terminal protocol message is buffered rather than committed immediately. The
scheduler requires protocol EOF and worker-leader exit, terminates the retained
group or Job Object to remove any surviving descendant, and only then commits
the terminal job/event transaction. Output after the buffered terminal or a
nonzero exit following a result becomes `WORKER_PROTOCOL_ERROR`; an expected
job-error or interruption frame retains its defined failed/interrupted outcome
after the same reconciliation.

The scheduler assigns immutable cursors. Event payloads follow the sealed
OpenAPI shapes. Per-type caps are checked before insertion: 2,001 metric, 2,001
checkpoint, six state, 32 phase, 32 warning, 16 persisted progress, and one
terminal event, within the 4,096 total cap. `after_cursor` is replayed ascending;
values beyond `last_cursor` return `SEMANTIC_INVALID`. SQLite triggers forbid
event update and deletion.

## Verification boundary

The focused source tests cover atomic acceptance and exact replay, rejection
without mutation, route scoping, queue saturation, queued cancellation,
reservation release, list-cursor filter binding, immutable event replay, restart
recovery, protocol identity/digest/order validation, finite typed events, the
closed unavailable handler table, terminal-frame/EOF/exit reconciliation,
cancellation-pipe failure, read-only recovery shutdown, and real WSL process
groups that are force-stopped without killing an unrelated sentinel, including
one whose leader has exited while a descendant remains alive.

Installed native Windows Job Object execution, installed fixed-argv defensive
worker rejection, all real operation handlers, browser behavior, and the sealed
605-case product-acceptance denominator remain `NOT_RUN` in this source slice.
