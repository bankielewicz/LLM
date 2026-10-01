# S0 AMD-056 — bind multi-role errors by role

Status: APPLIED to specification revision 1.2 under the owner's 2026-10-01 direction to proceed. This amendment resolves an S0 contract contradiction; it does not establish product execution or qualification.

## Finding

The revision-1.1 vocabulary stored one http_status, top_level_code and retryable tuple per code even when a code had multiple roles. PAYLOAD_TOO_LARGE is both a top-level transport code and a semantic reason. Its transport role is 413, while five existing semantic rules require 400 VALIDATION_FAILED with reason PAYLOAD_TOO_LARGE. The shared v1 tuple could not represent both behaviors.

## Amendment

Vocabulary format llm-foundations-error-vocabulary-v2 keeps direct fields for single-role rows and requires a bindings object whose keys exactly equal kinds for every multi-role row. Top-level and reason bindings carry concrete HTTP status. Reason bindings carry the matching top-level code. Job and terminal bindings carry null HTTP fields. All existing OpenAPI enums and wire behaviors remain unchanged. PAYLOAD_TOO_LARGE is bound as top_level=(413, null, false) and reason=(400, VALIDATION_FAILED, false).

The tuple order below is (http_status, top_level_code, retryable). The v1 tuple was shared structurally by every listed role; job-role HTTP fields were not emitted on the wire. V2 makes those non-HTTP fields explicitly null. The only corrected synchronous status is PAYLOAD_TOO_LARGE in its reason role.

| Code | Kinds | Revision 1.1 shared tuple | Revision 1.2 role bindings |
|---|---|---|---|
| ARCHIVE_UNSAFE | reason, job | (400, VALIDATION_FAILED, false) | reason=(400, VALIDATION_FAILED, false); job=(null, null, false) |
| CHAT_TEMPLATE_MISMATCH | reason, job | (400, VALIDATION_FAILED, false) | reason=(400, VALIDATION_FAILED, false); job=(null, null, false) |
| CHECKPOINT_INCOMPATIBLE | reason, job | (400, VALIDATION_FAILED, false) | reason=(400, VALIDATION_FAILED, false); job=(null, null, false) |
| CHECKSUM_MISMATCH | reason, job | (400, VALIDATION_FAILED, false) | reason=(400, VALIDATION_FAILED, false); job=(null, null, false) |
| CONTEXT_PREVIEW_STALE | reason, job | (400, VALIDATION_FAILED, false) | reason=(400, VALIDATION_FAILED, false); job=(null, null, false) |
| DATASET_INELIGIBLE | reason, job | (400, VALIDATION_FAILED, false) | reason=(400, VALIDATION_FAILED, false); job=(null, null, false) |
| DEVICE_UNAVAILABLE | top_level, job | (503, null, false) | top_level=(503, null, false); job=(null, null, false) |
| DISK_FULL | top_level, job | (503, null, true) | top_level=(503, null, true); job=(null, null, true) |
| INTERNAL_ERROR | top_level, job | (500, null, true) | top_level=(500, null, true); job=(null, null, true) |
| MODEL_NOT_PREPARED | reason, job | (409, STATE_CONFLICT, false) | reason=(409, STATE_CONFLICT, false); job=(null, null, false) |
| PAYLOAD_TOO_LARGE | top_level, reason | (413, VALIDATION_FAILED, false) | top_level=(413, null, false); reason=(400, VALIDATION_FAILED, false) |
| REFERENCE_MISSING | reason, job | (400, VALIDATION_FAILED, false) | reason=(400, VALIDATION_FAILED, false); job=(null, null, false) |
| SCHEMA_INVALID | reason, job | (400, VALIDATION_FAILED, false) | reason=(400, VALIDATION_FAILED, false); job=(null, null, false) |
| SEMANTIC_INVALID | reason, job | (400, VALIDATION_FAILED, false) | reason=(400, VALIDATION_FAILED, false); job=(null, null, false) |
| STORAGE_CORRUPT | reason, job | (503, STORAGE_UNAVAILABLE, false) | reason=(503, STORAGE_UNAVAILABLE, false); job=(null, null, false) |
| SUBJECT_INCOMPATIBLE | reason, job | (400, VALIDATION_FAILED, false) | reason=(400, VALIDATION_FAILED, false); job=(null, null, false) |
| TINY_CHECKPOINT_ALIAS_INVALID | reason, job | (400, VALIDATION_FAILED, false) | reason=(400, VALIDATION_FAILED, false); job=(null, null, false) |
| TOKENIZER_MISMATCH | reason, job | (400, VALIDATION_FAILED, false) | reason=(400, VALIDATION_FAILED, false); job=(null, null, false) |

## Validator changes

check_spec.py now rejects duplicate codes; invalid or duplicate kinds; missing or extra role bindings; flat fields on multi-role rows; bindings on single-role rows; non-null HTTP fields on job or terminal roles; missing synchronous statuses; reason/top-level status disagreement; and semantic-rule rejection tuples that disagree with the role binding. A built-in negative control confirms that 413 VALIDATION_FAILED / PAYLOAD_TOO_LARGE is rejected as a semantic response while the authoritative 413 top-level and 400 reason pair is accepted.

No requirement heading, trace case, acceptance execution unit, OpenAPI enum, semantic rule, or product status changes. All 605 product acceptance execution units remain NOT_RUN.
