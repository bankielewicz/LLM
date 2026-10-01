# S1 upload transport and archive-safety boundary

Status: implemented for the S1 candidate. Focused WSL development tests pass.
Product acceptance and the sealed AC-RUN cases remain `NOT_RUN`.

## JSON bodies

`receive_json` consumes the ASGI request stream without trusting chunk
boundaries. It rejects an invalid or mismatched `Content-Length`, stops at the
1 MiB transport limit before parsing, and accepts only well-formed
`http.request` messages with byte bodies and Boolean `more_body` flags. The
shared strict parser then enforces UTF-8, one JSON value, no duplicate object
keys, no non-finite numbers, no trailing data, and the sealed error vocabulary.
Schema and domain validation remain later service stages.

## Multipart staging

`receive_multipart` requires a declared length and one RFC-conforming boundary.
Part headers are capped at 16 KiB and 32 fields, folded headers and unknown
headers are rejected, and only `Content-Disposition` plus `Content-Type` are
accepted. Dataset uploads contain one JSON `metadata` part and one to three
unique `train`, `validation`, or `test` NDJSON parts. Bundle uploads contain
exactly one `bundle` part with media type
`application/vnd.llm-foundations.bundle+zip`. File-part media types are exact;
metadata may use the UTF-8 JSON charset parameter.

Client `filename` and `filename*` values are parsed only to close the header
shape and are never used as paths. Each file part is streamed to a random
server-generated mode-0600 staging name, hashed while written, flushed and
fsynced. A failed or cancelled parse removes only files created by that call.
Dataset files are limited to 10 MiB each and 30 MiB combined; metadata is
limited to 1 MiB. A bundle part is limited to 1 GiB. Bounded multipart framing
overhead is separate from those content limits.

The returned `StagedMultipart` orders parts by logical name. Its
`metadata_sha256` hashes canonical parsed JSON bytes, and its
`canonical_sha256` hashes canonical JSON of `metadata_sha256` plus sorted
`{name,size,sha256}` part records. Client filenames, session identity, and
wire-part order do not enter the digest, so the idempotency scope can remain
service instance, method, and resolved path.

## Inert ZIP inspection

`inspect_zip_archive` never extracts or registers content. It rejects an
archive above 1 GiB; comments, trailing bytes, multi-disk layout, encryption,
data descriptors, unsupported compression, unneeded extra fields, duplicate or
case-fold-colliding names, nonregular or executable entries, and declared or
actual expansion outside the 2 GiB/10,000-entry/512 MiB-per-entry/100:1 limits.
It fully reads every entry to verify CRC and actual sizes while keeping bytes in
the archive.

Names are strict portable POSIX paths: the manifest path or an object path under
`llm-foundations-bundle/`, ASCII `[A-Za-z0-9._/-]`, at most 240 UTF-8 bytes,
with no empty/dot/dot-dot segment, absolute/drive/UNC form, backslash, colon,
trailing dot/space, control form, or Windows device name. Manifest schema,
entry digests, dependency closure, provenance, and import eligibility are later
bundle-validation stages.

## Verification boundary

Focused tests cover chunk fragmentation, length mismatch, duplicate/non-finite
JSON rejection, order-independent multipart digests, ignored client filenames,
closed part names and media types, malformed/final boundaries, cleanup after
failure, bundle staging, traversal and nonportable names, links, executables,
duplicates, compression bombs, expansion caps, and trailing bytes. These are
source-level development checks. They do not establish HTTP integration,
disk-fault behavior, installed profiles, semantic dataset/bundle validity, or
the complete sealed acceptance matrix.
