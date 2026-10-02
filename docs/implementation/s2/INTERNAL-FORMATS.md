# S2 internal format and integration decisions

These choices fill implementation details left open by the sealed contract. They do not replace public schemas or qualify the product. The separate APP-009 proposal is an actual cross-contract conflict and remains outside these decisions.

## Dependency and source identity

`dependency_lock_sha256` is SHA-256 of the exact profile requirements file bytes. This matches the existing wheel manifest's `lock_sha256` binding. The generated `runtime_data/runtime-lock-manifest.json` separately records each wheel-manifest digest and is checked against all four frozen lock pairs.

The wheel build generates `runtime_data/build-provenance.json` inside the build output, without writing it into the source package. It binds Git commit, Git tree, a sorted digest/size inventory of all packaged source files, and whether the source was dirty. The final gate requires a clean committed source candidate and validates every installed package file against that wheel. An unbuilt source checkout advertises no executable lab capabilities; it cannot manufacture a source revision for checkpoint descriptors. Early development artifacts and final candidate artifacts remain distinguishable. At startup, an installed build must match its wheel location, non-cache package inventory, wheel-record sizes/digests, and recomputed packaged-source digest. A mismatch stops startup before storage creation.

## Checkpoint custody and worker acknowledgments

The service allocates each job's run/model/tokenizer identities and the bounded list of checkpoint IDs before dispatch. Only the service registers those identities. The worker consumes the leased checkpoint IDs in order so the incumbent can reference a new checkpoint before its manifest bytes are sealed.

The checkpoint wire message preserves DAT-010 exactly: `checkpoint_ready` carries `staging_name`, `manifest_sha256`, and seven `files` entries with `name`, `size`, and `sha256`. The entries include the manifest itself; its six payload entries are checked against the manifest. The service derives the next leased checkpoint ID from committed context and the step from validated trainer state. It acknowledges only after committing all seven artifact rows, the checkpoint ledger, run when first created, durable job boundary, and event in the same transaction.

Ordinary output proposals receive prepared artifact IDs without publishing registry rows. The terminal transaction commits those pending artifacts together with the typed result, run/model records, job state/event, and reservation release. This is distinct from the durable mid-run checkpoint acknowledgment.

## Record identities

For `evaluate`, the operation-specific `evaluation_id` is the parent-allocated `run_id`. The public schema defines no independent evaluation registry or requirement for different identities. This alias permits the evaluation result to be retrieved through the existing run registry without a new endpoint. Job identity remains distinct.

The checkpoint-file ledger maps each canonical file name to its immutable artifact ID, size, and digest. The run-artifact ledger records terminal output roles. Neither dependency closure nor resume resolution relies on the limited primary `artifact_ids` list.

## Tiny checkpoint config

The companion's closed internal config format is `tiny-v2-config-v1`, containing `format`, `architecture_profile_id`, `vocab_size`, `context`, `width`, `heads`, and `layers`. The exact bytes are bound by the checkpoint manifest and config digest. The loader validates the supported profile, dimensional bounds, head divisibility, and parameter cap before effective state construction. The public checkpoint/trainer/manifest schemas remain unchanged.

## Ordinary two-subject tiny comparison

RUN-012 permits one or two ordinary evaluate subjects and requires a paired result for a two-subject comparison. DAT-001 binds the existing `paired-evaluation.schema.json` specifically to the sealed/applied protocol; its applied-only fields cannot represent token NLL per byte. S2 therefore uses a separate closed internal `llm-foundations-tiny-paired-evaluation-v1` format for ordinary tiny comparisons. It records the common immutable dataset and ordered record IDs, the combined records artifact, both ordered tiny identities and byte-normalized totals, and candidate-minus-baseline NLL-per-byte delta. The operation result still uses the existing `paired_artifact_id` field. No statistical significance or cross-profile equality is claimed.

## Observed steps and terminal projection

For S2 runs, `step` counts completed optimizer updates observed by the parent. A fixed worker's readiness acknowledgment starts that observed count at zero; resume starts at the verified parent's update count. Nontraining runs request and perform zero optimizer updates, so their active/terminal count and terminal requested/final steps are zero. Queued and pre-spawn cancelled jobs retain null observed steps and create no run. This convention does not substitute zeros for missing metrics.

Training progress advances the observed count separately from the last durably committed checkpoint. The stored `run-result.schema.json` projection uses those exact parent fields and the same terminal timestamp as the job. Completed training must reach its requested final step; interrupted or failed jobs preserve the distinct observed and durable boundaries.
