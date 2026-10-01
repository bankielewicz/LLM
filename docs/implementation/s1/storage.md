# S1 storage, artifact, and dataset boundary

Status: implemented for the S1 candidate. Focused WSL development tests pass.
Native Windows control/preflight probes and the service integration suite are
separate observations. All 605 product acceptance cases remain `NOT_RUN`.

## Storage and recovery

`Database` is the serialized SQLite boundary. New roots receive schema version
1, a persistent installation UUID, foreign keys, WAL, and `synchronous=FULL`.
An existing database is first opened read-only: a future `user_version`, corrupt
bytes, a failed integrity check, or a missing/invalid installation identity
enters read-only recovery without changing the database or inventing a new
identity. Older schemas receive a validated timestamped backup before one
exclusive migration transaction. Mutation audit insertion is available both as
a caller-owned transaction operation and as a standalone transaction.

The root and all service-created subdirectories reject symlinks/reparse points.
POSIX files use no-follow opens and descriptor/path identity checks; the
instance lease also requires a regular file. Windows security calls declare
pointer-width-correct `ctypes` signatures, and current-user ACL inspection keeps
only the current user, SYSTEM, and built-in Administrators as permitted writers.
Artifact staging writes a random `.partial`, checks its bound, computes SHA-256,
flushes and fsyncs, then renames on the same root to
`objects/<prefix>/<sha256>/<sha256>` before inserting metadata. Existing bytes
are reused only after type, length, and digest verification. A database failure
after rename leaves an unreferenced object; no row ever commits before bytes.

Startup recovery verifies every unique registered object's path, regular-file
type, size, and digest. It moves unreferenced job/upload staging entries into
owned quarantine. Any bad reference or unsafe path enters read-only recovery;
it does not infer job completion, restore a backup, or delete evidence. Every
later artifact open verifies the same descriptor and digest using the opened
file before returning a stream positioned at byte zero.

## Capacity, identity, and lists

Reservations enforce the 53,687,091,200-byte root quota, the additional 1 GiB
free-space reserve, and dataset/job/run/model/checkpoint/artifact row limits
before acceptance. Held byte and row reservations participate in later
admission checks. Scheduler acceptance can create the canonical request
artifact and reservation inside its caller-owned job/event/idempotency
transaction; terminal transitions release that reservation in their own caller
transaction. Upload failures use the standalone release wrapper.

Artifact and registry lists order by `created_at` descending and UUID
descending. Their opaque HMAC-SHA-256 cursor binds the runtime instance, exact
schema, order, and filters. Decoding requires canonical unpadded base64url, so
alternate padding-bit encodings, tampering, a restart, or filter/order changes
return `VALIDATION_FAILED` / `SEMANTIC_INVALID` at `cursor`.

The authoritative bundled materialized-fixture registry supplies the sealed
capstone digest. Ordinary artifact content reads reject that digest regardless
of which logical artifact or split refers to the shared bytes, with
`STATE_CONFLICT` / `SEALED_SPLIT_REQUIRES_TOKEN`. Metadata remains listable.
Only a later trusted worker path may request an internal read after token
consumption; S1 exposes no bypass through HTTP.

## Dataset registration

`DatasetRegistry` accepts only the validated name/record-format/provenance
metadata and one to three named streams. It stages each stream under its upload
UUID, enforces 10 MiB per split, 30 MiB combined, and 100,000 records, then
applies the required validation order: UTF-8, JSONL syntax, closed record shape,
cross-record identity, checksum through staging, semantic invariants, and final
eligibility. `retrieval_v1` accepts only `train`; instruction records require
the canonical one-user-message and `INTENT`/`REPLY` form.

The `data-audit-v2` implementation compares exact UTF-8 content across splits,
performs the specified NFKC/casefold/punctuation/code-token/digit/whitespace
near normalization, and counts unordered cross-split pairs without quadratic
pair enumeration. It separately counts group IDs crossing splits, split and
slice records, and exact text-field UTF-8 bytes. Any exact leakage, normalized
near leakage, or group overlap produces `audit_only`; clean input is `eligible`.

Registration allocates server IDs and computes `manifest_sha256` only from
record format plus split digest/count/file-byte/sealed values. Name,
provenance, local IDs, and times cannot change this content identity. Split and
audit artifacts, dataset and split rows, database revision, reservation
release, and the exact stored 201 idempotency response commit in one SQLite
transaction. Shipped provenance is granted only when the supplied split map
matches an authoritative bundled release family; the sealed source must retain
the logical `test` role.

## Verification boundary

Focused tests cover read-only-first future-schema handling, durable SQLite
settings and installation identity, lease link rejection, atomic object reuse,
verified reads, canonical authenticated cursors, capacity rejection and
release, job request rollback, staging quarantine, corruption recovery, all
five authored dataset families, the clean and leaky audit oracles, content-only
manifest identity, rejection without registry revision change, retrieval role
enforcement, and sealed-content denial. These are source/service development
checks. They do not establish browser behavior, installation qualification,
multi-profile crash behavior, owner acceptance, or any sealed product AC case.
