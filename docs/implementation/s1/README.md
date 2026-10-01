# S1 local companion

This work slice adds the local service around the revision 1.2 contracts. It starts from S0 merge commit `6fc92e4abab1146da9fc9d969d49d8733dbd93e0` and preserves the 150 sealed specification files and all 107 protected source files. The S0 authority receipt remains the authority for those inputs; this document does not amend them.

## Delivered surface

The wheel serves the preserved reader and a local pairing shell on `http://127.0.0.1:8765`. The shell removes the pairing fragment before loading the reader or sending a request, exchanges the single-use grant, and holds credentials only in instance-scoped session storage. HTTP authentication, private owner IPC, root ownership, transactional dataset and artifact registration, strict errors, pagination, the one-worker scheduler framework, immutable events, cancellation and startup recovery are separate components.

Dataset registration independently validates every record, computes leakage counts and eligibility, and registers all split artifacts and the audit together. Bundle upload registers an inert, transport-safe ZIP artifact. Uploading a bundle does not validate its application semantics or import its contents. The service never imports a model framework; the preflight and each admitted job use separate owned child processes.

Every production model and bundle-operation handler remains unavailable in S1. A valid job request receives `503 CAPABILITY_UNAVAILABLE` before a job is created. Missing or mismatched backend dependencies also leave the CPU reader available with unavailable capabilities. S2 supplies the tiny-model operations, S3 the intermediate learner interface, and S4 the applied-model operations. The preserved reader does not acquire live training controls from this slice.

## Run the development slice

Use a Python 3.12 environment prepared from the matching complete runtime lock in `companion/locks/`. Keep that environment separate from the development-test environment: their pinned build-tool dependencies differ. Build the companion wheel with the development environment, then install only that wheel with `--no-deps` into the prepared runtime environment. Do not install it into the global Python environment.

```text
python -m llm_foundations_companion gpu-check
python -m llm_foundations_companion serve --storage <new-storage-root>
python -m llm_foundations_companion connect --storage <storage-root>
python -m llm_foundations_companion status --storage <storage-root>
python -m llm_foundations_companion request --storage <storage-root> --file <request.json>
python -m llm_foundations_companion cancel --storage <storage-root> --job-id <uuid>
python -m llm_foundations_companion events --storage <storage-root> --job-id <uuid> --after <cursor>
```

`serve` prints the resolved root and a pairing URL. Open that URL in a local browser. The port is fixed; an occupied port or owned root fails without choosing another location. `connect` replaces the outstanding two-minute grant without interrupting jobs. Owner commands use private local IPC and never save browser credentials. The optional request argument `--idempotency-key <uuid>` permits an explicitly identified retry.

On WSL, the fixed `<root>/runtime/control.sock` pathname must fit within 107 UTF-8 bytes; choose a shorter root if startup reports that limit. The service does not select another IPC path.

Stopping the service resolves its owned worker before releasing the root lease. Restart creates a new instance and invalidates every browser session. Stored artifacts and datasets survive. The same root can be reopened with a different qualified profile only after a clean stop. A future or corrupt database is not repaired destructively: recovery keeps available reader/runtime diagnostics and rejects writes. Preserve the root and its retained backups for investigation.

Uninstall the package or remove the isolated installation environment after stopping the service. This does not delete the selected storage root. Keep or back up that root independently; no uninstall command removes learner data.

## Validation and scope

`python scripts/run_s1_gate.py --help` describes the complete source/service gate. It requires the exact S0 development environment, the prepared WSL CPU environment, the separate legacy-regression environment, Node, and a native Windows Python 3.12 interpreter for the named-pipe control probe. Every invocation requires a new report directory outside the source worktree. Failed attempts, blocked prerequisites, raw process output, source manifests and built wheels remain there. Reports include commit, tree, source-manifest digest, wheel digest, interpreter identity and output hashes.

The gate runs the historical S0 test files explicitly, then the complete current companion suite. The WSL report retains its native-Windows-only pytest case as `NOT_RUN`; a separate required native Windows IPC process probe reports its own result. Neither replaces the other. Installed-wheel checks bind the interpreter, wheel, package members and exact distribution `RECORD`; reject package or distribution-metadata symlinks and unexpected files; and launch the installed fixed `-I` worker through `ready`, `WORKER_PROTOCOL_ERROR` and EOF while confirming all 15 handlers remain unavailable. Real WSL HTTP and IPC checks exercise pairing/revocation, dataset registration, exact idempotent replay, malformed-upload atomicity, artifact downloads, inert bundle upload, unavailable-job rejection, exclusive launch and restart persistence. Private protocol fixtures exercise scheduler boundaries and owned-process cancellation; they do not implement production model operations.

The frozen product acceptance registry still contains **605 execution units, all initially `NOT_RUN`**. This work does not mark them passed. A development-wheel smoke check in the Codex in-app browser observed successful pairing, a clean visible URL, pairing across reload, disconnect with the reader retained, and the disconnected state across reload; no warning or error console entries were captured. This observation does not qualify the final candidate or the broader browser requirements.

Full browser acceptance, training/evaluation behavior, all four native profile qualification campaigns, fit calibration, release distribution and owner acceptance remain separate obligations. A successful S1 source/service gate establishes only the checks recorded in that report.
