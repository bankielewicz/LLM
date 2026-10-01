# Cross-section integration contract

Status: NORMATIVE TARGET, specification revision 1.1 (approved by the owner on 2026-09-30). This document binds the joins between learner actions, runtime requests, artifact identity, and acceptance. It does not report product execution. Conflicting prose/schema/fixture bytes are a specification defect; they are not a choice offered to an implementer.

## INT-001 — Bind each operation to its effects
The reader MUST use the following operation distinctions everywhere, including setup, review, monitor, results, notes, and export.

| Learner action | Operation | Reads | Changes | Durable evidence |
|---|---|---|---|---|
| Train tokenizer | tokenizer_train | eligible train text | merge rules only | tokenizer JSON/digest and creating job |
| Train TinyLM | tiny_train | fixed tokenizer, eligible train/validation | new random-initialized model and optimizer | run, metrics, safe checkpoints |
| Resume TinyLM | tiny_resume | safe model/optimizer/RNG checkpoint | child model/optimizer | parent link and child run |
| Evaluate | evaluate | selected immutable subjects and split | measurements only | protocol, complete record IDs, raw outputs and aggregates |
| Prepare model | model_prepare | pinned safe remote bytes | verified local cache/registry | download manifest and model ID |
| Train adapter | adapter_train | frozen base and train/validation | LoRA parameters and optimizer only | adapter run/checkpoints and frozen-base checks |
| Resume adapter | adapter_resume | base plus complete adapter resume state | child adapter/optimizer | parent link and restored-state evidence |
| Preview model context | context_preview | tokenizer/template, subject identity, proposed request | preview artifact only | exact serialization, crop/drop decisions and request digest |
| Prompt and continue | generate | tiny checkpoint and matching preview | output artifact only | prompt, new token IDs/text, seed and checkpoint |
| Send chat message | chat_generate | base/adapter and matching preview | output artifact only | serialized context, generated tokens and subject |
| Build/query retrieval | retrieval_build / retrieval_query | corpus/index | immutable index or ranked-hit artifact | TF-IDF identity and exact retrieved IDs |
| Export / validate / import | export_bundle / validate_bundle / import_bundle | selected objects/archive | copy / validation record / imported logical identities | manifest, validation facts and import map |

Conversation history is committed by its explicit revision-checked API action after generation; generating alone does not mutate a conversation. Neither history commit nor context preview trains a model. A legacy .pt record is not a portable v2 checkpoint.

## INT-002 — Join screens, APIs, and durable state
The OpenAPI document MUST contain a callable path for each local action below; the UI MUST NOT invent a successful local state before its response is committed.

| Surface/action | Contract boundary | Persisted state |
|---|---|---|
| Pair/reconnect | sessions; runtime; preflight | session in sessionStorage only; preflight receipt |
| Data Source/Inspect/Split | strict browser inspection; fixed group allocation | unsent form state only |
| Save audited dataset | POST datasets with metadata and split files | manifest, audit, split objects; audit-only allowed |
| Review/submit/cancel job | jobs; cancellation; events/snapshot | immutable request, state events, typed result |
| Select tokenizer/checkpoint | tokenizer/model/checkpoint registries | exact ID and digest, never display-name identity |
| Preview then generate | context_preview then generate/chat_generate | preview and output bound to identical effective request |
| Save progress | progress module update with revision | independent dimension and timestamp |
| Save/delete note | notes with revision; deletion preview/confirmation | revision history or tombstone |
| Save chat history | conversations and generation commit with revision | subject-bound turns and generation artifact IDs |
| Export chat as data | explicit conversation export | inert dataset-candidate artifact, later ordinary import/audit |
| Verify evidence | evidence verification profile | immutable digest-bound facts, not prose grade |
| Assess reasoning | evidence assessments | identified review kind, rubric rows and derived decision |
| Begin/finalize capstone | capstone attempt; paired evaluate token | freeze, exposure ledger, consumed state, paired artifact |
| Backup/import learning state | backup export; progress import preview/apply | explicit selected backup and atomic merge |
| Reuse model bundle | bundle upload; validate_bundle; import_bundle | imported IDs and preserved source identity |



```mermaid
flowchart LR
  S["Hosted/static reader"] --> B["Browser progress and imported claims"]
  S --> X["Explicit backup export"]
  X --> I["Learner-selected import"]
  L["Local reader · 127.0.0.1:8765"] --> A["Authenticated same-origin API"]
  I --> A
  C["Owner CLI"] --> P["Private control channel"]
  P --> A
  A --> Q["FIFO scheduler · one worker"]
  Q --> W["Fixed operation worker"]
  W --> O["Immutable objects and typed results"]
  A --> D["SQLite metadata and audit"]
  O --> D
  D --> E["Snapshots and replayable events"]
  E --> L
  H["Explicit E01 learner-source CLI"] --> V["Hash-bound verification receipt"]
  V --> P
```

The diagram has no hosted-reader-to-loopback execution edge. E01 source execution remains in its explicitly invoked CLI; the worker executes only installed operation modules.

Browser inspection is preliminary. Final dataset eligibility comes from the companion's revalidation of uploaded bytes. Group splitting is explicit, deterministic, and represented in the submitted split files; a browser cannot mark an unaudited selection eligible.

## INT-003 — Define the TinyLM v2 mechanics
tiny-v2-standard-v1 MUST preserve the untied architecture, initialization, forward/target contract and pure byte/BPE algorithm of the protected [model.py](../../../course/labs/model.py), [data.py](../../../course/labs/data.py), and [tokenizer.py](../../../course/labs/tokenizer.py). These source bytes are bound by SCP-001. The companion refactor changes custody/instrumentation and adds named protocols; it MUST NOT silently replace attention, LayerNorm, GELU, learned positions or the optimizer.

The standard model has token embedding V by W, learned position embedding C by W, L pre-norm blocks, biased QKV and attention-output projections, width-4W GELU MLP, final LayerNorm, and an untied biasless output W to V. There is no dropout, rotary position, RMSNorm, cache or mixed precision. Unique trainable scalar count is 2VW + CW + L(12W² + 13W) + 2W. Tied variant subtracts VW and follows APP-014. The default lesson configuration is V=257/C=64/W=64/H=4/L=2, batch 8, learning rate 0.0003, seed 17; a preset overrides only explicitly named fields.

Tokenizer training reads train only in stored record order. Vocabulary request 257 returns the byte tokenizer with zero merges; greater values use protected deterministic pair counts/ties, stopping early if no pair has count 2. Actual vocabulary may be below the requested size and MUST be reported. tokenizer_type is `byte` exactly when the trained tokenizer has zero merges (byte-v1, byte-bpe-v1 at vocab_size 257, or no pair reaching count 2) and `byte_bpe` otherwise. For tiny tokenizers tokenizer_sha256 is the protected `Tokenizer.fingerprint()` value (SHA-256 of `json.dumps(to_dict(), sort_keys=True)`); tokenizer.json bytes are exactly `json.dumps(to_dict(), indent=2)` without a trailing LF, as the legacy lab writes them, and their file digest is the artifact sha256, a different value. The 200,000-byte cap counts the strict UTF-8 bytes of the train split's text fields only (legacy lab.py L99-L100) for tokenizer_train, tiny_train and tiny_resume. Seed is retained as request metadata but does not alter this deterministic BPE algorithm. EOS is 256; merges start 257; fingerprint uses the protected tokenizer's exact serialization, not a new compact-JSON digest.

Training concatenates each ordered record's token IDs followed by EOS, independently for train/validation, and requires each stream longer than C. A CPU Torch generator seeded with the run seed samples starts uniformly exactly as protected batch(). Targets are shifted once by the loader. AdamW uses lr from request, betas 0.9/0.999, eps 1e-8, weight_decay 0.01; zero_grad(set_to_none=True), scalar mean token cross-entropy, backward, global L2 gradient clipping 1.0 with error_if_nonfinite=True, optimizer step. No scheduler, warmup, accumulation, augmentation or shuffle beyond sampled starts is added.

Before initialization, set Python random seed and torch.manual_seed, Torch intra-op and inter-op thread counts to 1, deterministic algorithms on; CUDA sets CUBLAS_WORKSPACE_CONFIG=:4096:8 before Torch import and disables TF32/cudnn benchmarking. An unsupported deterministic operation fails explicitly. Record exact device/profile/lock; no cross-profile equality claim.

Training metrics at initial step 0, each global update divisible by eval_every, and final update use the protected eight fixed sampled batches per split with seed+1000. Resumed initial measurement at the saved global step does not update weights or consume batch RNG. Select lowest validation token NLL, tie earliest global step; inherit the parent's incumbent. Train and validation values retain units nats/token and MUST NOT be mixed with the separate byte-normalized protocol. Each update's gradient diagnostics are retained in the bounded metric artifact; the event stream follows RUN-011 cardinality rather than emitting an unbounded event per diagnostic field.

## INT-004 — Serialize TinyLM checkpoints without pickle
Tiny-v2 checkpoint payload MUST contain model.safetensors, optimizer.safetensors, rng.safetensors, tokenizer.json, config.json, trainer_state.json and manifest.json. Manifest lists every other file with exact size/SHA-256 and format tiny-v2-checkpoint-v1. No executable or pickle format is permitted.

model.safetensors contains named state_dict tensors, including attention mask buffers; the tied variant writes shared storage once and declares the APP-014 alias. optimizer.safetensors stores exp_avg and exp_avg_sq under stable parameter names; trainer_state.json stores each parameter's integer AdamW step, ordered parameter groups/settings, completed global step, requested final step, eval_every, batch size, seed, incumbent best checkpoint ID/loss/step, dataset split identities, tokenizer digest, architecture profile/variant, profile/device and lock digest. rng.safetensors stores batch CPU generator, Torch CPU generator and per-selected-CUDA-device RNG byte tensors with explicit names. Python RNG state, if used, is integer-only JSON with exactly the version/state/gauss representation, never executable deserialization.

Load MUST validate files, tensor names/shapes/dtypes, aliases and identities before constructing effective model/optimizer state. Unknown/missing tensors, a nonfinite incumbent, wrong dataset/tokenizer/config/lock/profile, or unsupported version rejects load with CHECKPOINT_INCOMPATIBLE (400 VALIDATION_FAILED before queueing when registry metadata already shows the mismatch; otherwise the job fails with that code before any tensor is used). Parent step plus additional_steps above 2,147,483,647 is 400 VALIDATION_FAILED reason TINY_STEP_OVERFLOW. Tiny nonfinite loss/gradient fails the job with NONFINITE_TRAINING_VALUE after retaining the last safe checkpoint; an unsupported deterministic operation fails with DETERMINISTIC_KERNEL_UNAVAILABLE; a train or validation token stream not longer than the context fails with SPLIT_TOO_SHORT. Initial incumbent is represented by null before the first validation; infinity is never JSON. Validation loss is finite after initial evaluation.

Each saved boundary receives a new checkpoint ID. latest and best are registry references, not mutable files.

Tiny checkpoints are committed at global step 0 of a fresh tiny_train (after the initial evaluation), at every global step divisible by eval_every, at the requested final step, and at the cancellation boundary; `training_identity.save_every` equals `eval_every` (semantic rule TINY_SAVE_CADENCE). Every tiny evaluate, generate and context_preview subject is a checkpoint (`checkpoint_id`); a tiny model record is only a listing handle created at the run's terminal commit when the run committed a checkpoint, and its `checkpoint_id` is the run's last committed checkpoint. TinyTrainResult and TinyResumeResult return `best_checkpoint_id` (lowest validation token NLL across the lineage's committed checkpoints, earliest global step on ties, parent incumbent inherited), `last_checkpoint_id` and `metrics_artifact_id`. Interrupted runs expose their committed checkpoints through checkpoint_boundary and GET /checkpoints. The checkpoint digest (`sha256` in the descriptor, `checkpoint_sha256` in subject identities and ModelDetail) is the SHA-256 of the checkpoint's exact `manifest.json` bytes, which validate against [checkpoint-manifest.schema.json](contracts/schemas/checkpoint-manifest.schema.json) and list every other file's name, size and SHA-256; `trainer_state.json` validates against [tiny-trainer-state.schema.json](contracts/schemas/tiny-trainer-state.schema.json). An inference-only export writes a manifest listing only model.safetensors, tokenizer.json and config.json, so it is a different checkpoint identity with its own digest; its source_identity records the original checkpoint digest. Tensor names are fixed: model.safetensors uses the protected state_dict names (`tokens.weight`, `positions.weight`, `blocks.<i>.norm1.weight|bias`, `blocks.<i>.attention.qkv.weight|bias`, `blocks.<i>.attention.output.weight|bias`, `blocks.<i>.attention.mask`, `blocks.<i>.norm2.weight|bias`, `blocks.<i>.mlp.0.weight|bias`, `blocks.<i>.mlp.2.weight|bias`, `norm.weight|bias`, `output.weight` - absent in the tied variant, whose alias `{output.weight: tokens.weight}` is stored in manifest.json `aliases`); optimizer.safetensors uses `<parameter name>.exp_avg` and `<parameter name>.exp_avg_sq` for each unique parameter; rng.safetensors uses `batch_cpu`, `torch_cpu` and `cuda.<device index>`. Resume inherits all state except additional_steps; device/profile changes are rejected. Global final step equals parent step plus additional_steps and cannot exceed 2,147,483,647. A 25+25 resumed control MUST match an uninterrupted 50-update run's tensors, batch RNG, fixed validation values and greedy token IDs on the same qualified profile. Times, paths, job IDs and artifact IDs are intentionally different.

E01's explicit learner-source CLI is distinct from the fixed worker. A successful hash-bound verification receipt permits the built-in tied training profile and is referenced by its run; the web worker never executes the learner's patch. This verifies the submitted modification and separately trains the equivalent specified built-in variant. No operating-system sandbox for learner code is claimed.

## INT-005 — Define complete byte-normalized evaluation
tiny-nll-per-byte-v1 is a teacher-forced full-split protocol, distinct from the eight-window training monitor. Sort records by record_id in Unicode code-point order. For a record with tokenization [t0,...,t(m-1)], begin history with EOS256. For each target in [t0,...,t(m-1),EOS256], feed at most the final C history tokens to model.eval() under inference_mode, compute log_softmax of the final position's logits in float64, accumulate negative log probability of the target, then append that target to history. Learned positions restart at 0 for each cropped model call. There is no cross-record history, sampling, skipped first token, batching-dependent padding or target truncation. tiny-nll-per-byte-v1 accepts a split of at most 200,000 strict UTF-8 text bytes (semantic rule EVALUATE_TINY_TEXT_BYTES), bounding its one-forward-pass-per-target cost; it polls cancellation after each record. Adapter training polls cancellation per APP-007 (after each microbatch and before each optimizer step), which governs over RUN-009's per-update wording for adapters.

utf8_bytes is the strict UTF-8 length of original record text, excluding JSON framing and EOS. token_count is m+1 and includes the scored EOS. The denominator is bytes, not token_count. Empty/whitespace-only records are rejected during registration. Report each finite NLL, byte count, ratio, and the aggregate sum(NLL)/sum(bytes), not the mean of record ratios. Per-record and aggregate sum order is fixed record/target order; float64 accumulation tolerance is absolute 1e-12 for fixture arithmetic and absolute 1e-6 for same-profile model comparisons. EOS scoring is part of this protocol; the value is not claimed to be a universal benchmark.

The arithmetic fixture is records A: NLL 6/bytes 3, B: NLL 4/bytes 8: aggregate 10/11, while the mean of ratios 1.25 is incorrect. A uniform V=257 logits fixture scores each target at ln 257; a one-byte record containing one byte token plus EOS has NLL 2 ln 257 and byte denominator 1. BPE comparison requires same original record bytes and this identical EOS/history protocol.

## INT-006 — Freeze preview and decoding semantics
Every interactive generate/chat request MUST reference its matching context-preview artifact. Context preview executes tokenizer/template work in the fixed worker, uses the same queue/authentication/bounds, and never loads or changes learned weights. Its canonical-request digest excludes preview_artifact_id and context_preview_digest and binds every subject identity, prompt/message byte, decoding field and effective profile. The service checks request/artifact/registry identities before queueing without tokenizing. The accepted worker re-tokenizes and verifies the complete preview before inference; mismatch fails the job with CONTEXT_PREVIEW_STALE. subject_identity_sha256 is the SHA-256 of the same canonical JSON encoding over exactly {backend, model_id, checkpoint_sha256, runtime_profile, device, context_limit}; checkpoint_sha256 is null for a prepared base and, for tiny or adapter subjects, the SHA-256 of the checkpoint's exact manifest.json bytes (INT-004). The profile and limit therefore remain bound even though they are resolved from registry metadata rather than client fields. Evaluation uses its frozen protocol internally and requires no interactive preview.

Tiny continuation encodes the nonempty prompt without EOS and previews the final C tokens used for the first step, original count and left-cropped count. Subsequent steps feed the final C tokens of prompt plus already generated tokens. Prompt bytes are displayed separately from new output. Tiny top-p is greater than 0 through 1; temperature 0 requires top-p 1. Positive-temperature decoding divides logits by temperature, sorts probabilities descending with token ID ascending for ties, retains the smallest prefix whose cumulative probability is at least top_p, renormalizes, and samples with a job-local generator. Greedy chooses lowest ID among exact maximum logits. EOS stops before adding it to visible text but remains in generated token IDs/count. Stop reason is eos, length or cancelled; decode uses UTF-8 replacement for invalid generated byte sequences and reports replacement occurrence. Input tokenization round-trip uses strict decoding. Tiny decoding casts logits to float64, divides by temperature and computes softmax and cumulative sums in float64; chat uses the APP-009 float32 softmax with the cumulative sum computed in float64. Both sample with `torch.multinomial(probabilities, 1, generator=job_generator)` where `job_generator = torch.Generator(device=selected_device).manual_seed(seed)`. Temperature is 0 or within [0.01, 2]. EOS is included in the generated token IDs and generated_token_count for both backends; when the max_new_tokens-th token is EOS the stop reason is eos. [fixtures/applied/decode-oracle-v1.json](fixtures/applied/decode-oracle-v1.json) freezes sort order, retained sets, greedy ties and stop cases. Tiny preview fields: original_input_token_count is the prompt's token count; input_token_ids are its final min(C, count) tokens; cropped_input_tokens is original minus final; effective_context_budget is C; serialized_text is `decode(input_token_ids, errors="replace")`, which begins with U+FFFD when the window starts inside a multi-byte character. Chat preview fields: original_input_token_count counts the full serialized transcript before any drop; cropped_input_tokens is that count minus the retained count; effective_context_budget is 512 minus max_new_tokens. The subject identity `backend` value is the preview vocabulary `tiny` or `chat`.

Applied context/template/decoding uses APP-004 and APP-009/010. No browser estimates pretrained token count by character length. Preview UI labels executed tokenizer work distinctly from model inference. All matching preview artifacts remain provenance records even if the learner never sends the generation.

## INT-007 — Enforce numeric limits at their actual boundary
All limits are inclusive. KiB/MiB/GiB mean powers of 1024. UTF-8 byte checks occur after strict decoding but before acceptance; Unicode scalar counts reject lone surrogates. JSON Schema maxLength is a scalar bound, not proof of a byte bound. Enforce the byte rule separately. Every limit or relation that JSON Schema cannot express is a rule in [contracts/semantic-rules.json](contracts/semantic-rules.json) with its reason_code and field_path, and each request rule is listed in its OpenAPI component's `x-semantic-rules`. At limit+1 reject before job/registry mutation unless a row explicitly defines display truncation.

| Boundary | Maximum or fixed bound | Rejection/result |
|---|---|---|
| Ordinary API JSON body |1MiB bytes; no content encoding |413 PAYLOAD_TOO_LARGE |
| Dataset upload |3 split files;10MiB each;30MiB combined;100,000 total records |413 PAYLOAD_TOO_LARGE; zero registration |
| Tiny/tokenizer parsed training text |200,000 UTF-8 bytes of the train split's text fields |400 VALIDATION_FAILED, reason PAYLOAD_TOO_LARGE |
| Applied parsed SFT content |20MiB UTF-8 bytes |400 VALIDATION_FAILED, reason PAYLOAD_TOO_LARGE |
| Tiny architecture | context 8..512; width 16..512; heads 1..16; layers 1..12; batch 1..64; P <= 10,000,000 |400 VALIDATION_FAILED |
| Tiny updates |1..2000 per job; global step<=2,147,483,647 |400 VALIDATION_FAILED |
| Adapter updates |1..120 per job and lineage cumulative<=120 |400 VALIDATION_FAILED, reason ADAPTER_UPDATE_LIMIT |
| Tiny prompt/output |16,384 UTF-8 input bytes;1..512 new tokens |400 VALIDATION_FAILED |
| Chat request |1..64 messages;8192 bytes each;32768 content bytes total;1..128 new tokens |400 VALIDATION_FAILED |
| Applied effective context |512 including generated-token budget |context_preview job fails with job.error.code CONTEXT_TOO_LONG; no preview artifact, so no generation can be requested |
| Retrieval | query 4096 UTF-8 bytes; top_k 1..20; max_features 100..20000 |400 VALIDATION_FAILED |
| Text artifact preview |262,144 bytes |larger safe text is download-only; no clipped pseudo-complete preview |
| Notebook |5 fields <=20,000 scalars each;8 links each module/dataset/run/checkpoint;16 artifact links |400 VALIDATION_FAILED |
| Evidence/assessment text |32 artifact links;claim/rationale 20,000 scalars;reviewer ID/name 120 |400 VALIDATION_FAILED |
| Assessments per evidence |10 |409 REGISTRY_LIMIT, reason REGISTRY_COUNT_EXCEEDED |
| Conversation storage |200 turns;1MiB total UTF-8 content |409 REGISTRY_LIMIT; current revision unchanged |
| Learning-state JSON | v2 wrapper 8 MiB; plain progress 1 MiB; legacy v1 backup retains the preserved decimal 25,000,000-byte limit (recorded-run files keep 1,000,000 bytes combined) |413 BACKUP_TOO_LARGE on export;413 PAYLOAD_TOO_LARGE on import |
| Static sidecar | 4 MiB measured as 2 * serialized JS string.length; 100 live notes |local save rejects with storage message; existing bytes unchanged |
| Bundle |1GiB archive;2GiB expanded;10,000 entries;512MiB/entry;240-byte path;100:1 ratio |413 PAYLOAD_TOO_LARGE at upload; structure/ratio/path failures fail the validate_bundle or import_bundle job with ARCHIVE_UNSAFE |
| Job queue |1 active worker;8 queued |429 QUEUE_FULL |
| Worker deadline/cancel |60min from starting;30sec cooperative shutdown |interrupted/timeout; latest durable boundary only |
| Teaching hold |step 25;10min |interrupted/exercise_timeout on expiry |
| Job events |4096; closed cardinalities in RUN-011 |WORKER_PROTOCOL_ERROR with terminal reserve |
| Worker stdout/stderr |1MiB each |retain prefix; explicit truncation warning and byte count |
| Service logs |10MiB/file;5 retained files |rotate oldest diagnostic file; never prune job/evidence records |
| Paging |100 list items;500 event rows |400 VALIDATION_FAILED |
| Root storage |50GiB committed+staging+reserved new bytes |409 REGISTRY_LIMIT, reason ROOT_QUOTA_EXCEEDED, before mutation |
| Free space at acceptance |estimate + 1GiB |503 DISK_FULL, reason INSUFFICIENT_STORAGE, retryable, before any job row |
| Registry counts |1000 datasets;50,000 jobs;20,000 runs;512 models;5000 checkpoints;100,000 artifacts |409 REGISTRY_LIMIT |
| Learner registry |1 progress document/24 allowed modules;1000 live notes;5000 evidence;200 live conversations;100 capstone attempts;20 active import/deletion previews |409 REGISTRY_LIMIT, reason REGISTRY_COUNT_EXCEEDED |

Free-space reserve is 1 GiB above the estimated maximum new bytes, and root quota includes reservations for queued jobs. Tiny train/resume reserve max(512MiB, Ck*(12P+8MiB)+16MiB), where Ck=2+ceil(requested additional updates/eval_every), including initial and final-or-cancellation checkpoint allowance. The 8 MiB per-checkpoint overhead covers maximum attention-mask buffers, JSON metadata and RNG/optimizer scalars. Cancellation at an already durable update reuses that checkpoint and emits no duplicate checkpoint-committed event. P is unique trainable scalars; tied count excludes aliases. Adapter reserve max(1GiB, Ck*(12*460800+4MiB)+64MiB), with Ck=2+ceil(updates/preset_cadence), cadence 10 for intents and 5 for capstone. Remaining operation estimates are RUN-007. Acceptance also reserves checkpoint, artifact, run and model registry rows as specified in RUN-007, and a request whose own byte estimate exceeds the root quota is 400 VALIDATION_FAILED reason ESTIMATE_EXCEEDS_ROOT_QUOTA. A reservation is released only after terminal commit/recovery; free-space deterioration can still fail during execution. A measured job's speed does not change any cap automatically.

## INT-008 — Use one error vocabulary and validation order
Synchronous errors use RUN's envelope with uppercase code and optional uppercase reason_code. Transport/authentication errors retain their own status. Semantic/parser/domain rejections use HTTP 400 code VALIDATION_FAILED plus the exact DAT/APP reason. Resource identity/state conflicts use 409 STATE_CONFLICT. A post-accept domain failure uses that uppercase domain cause directly as job.error.code; it is not reported as a synchronous rejection. The closed vocabulary is [contracts/error-vocabulary.json](contracts/error-vocabulary.json): each code's kind (top-level, reason, job, terminal), stage, HTTP status, top-level code and retryability are defined there once, and the OpenAPI `ReasonCode`, `JobErrorCode` and `TerminalReason` enums are generated from it. A code absent from that file MUST NOT be emitted. A schema-stage failure reports reason_code SCHEMA_INVALID, except that when every failing keyword lies inside a subschema annotated `x-reason-code`, that annotation supplies the reason (for example INVALID_SAMPLING_PARAMETERS for the greedy top_p rule).

DAT's order is transport size, archive structure when applicable, encoding/JSON, schema, reference, digest, semantic invariant, eligibility. Authentication/Host/Origin/CSRF checks precede that order. Return lexically sorted field errors from the first failed stage; do not expose source text/host paths or invoke later expensive model work. A rejected acceptance request has no job, no accepted-state event and no registry revision change. An inert staging upload/preview is a separately named accepted operation and cannot be described as a zero-write validation.

Canonical request bytes use the qualified Python 3.12 call json.dumps(value, sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False).encode("utf-8"), with no trailing LF and no Unicode normalization. Validate types first, then re-serialize by schema type: integer-typed fields are integers (a JSON number with a fraction part or exponent in an integer field, such as 1.0, is rejected with SCHEMA_INVALID), number-typed fields are binary64 floats (so 1 and 1.0 canonicalize identically as 1.0), and strings are preserved exactly. Service-generated request digests use this serialization consistently across preview, acceptance and idempotency; the frontend reuses the saved preview request object. Bundle manifest canonicalization retains DAT-009's separate trailing-LF rule.

Preview/create/apply tokens bind input digest and database/entity revision. Rejection does not make a stale preview valid. Repeating an accepted request with the same idempotency key and bytes returns its prior result; changing bytes under that key returns IDEMPOTENCY_CONFLICT. A client recovering across session expiry first reconciles the prior job/request, never auto-resubmits uncertain work.

## INT-009 — Keep progress, assessment, and test exposure independent
The 24 allowed module IDs are exactly P00,00..20,E01,E02. Progress has independent reading/read/practiced/self_checked flags; artifact verification and review records are separate. review_kind is exactly self or independent. An independent review requires a named human reviewer; the app records that assertion, not verified employment/identity. Imported assessments remain imported claims until a new local assessment is recorded. A rubric decision is derived from its required rows; clients cannot grant meets by sending only a total.

Static mode preserves the foundation v1 storage and uses the named intermediate sidecar for added progress/notes/claims. Migration is explicit and retains source bytes. Static code cannot execute companion artifact verification or award local execution attestation. Companion records remain authoritative for real jobs; origins exchange selected backups/bundles only.

A capstone token is consumed before test loading. The dataset-digest exposure ledger records every consumed attempt, including failure/cancellation. A replacement attempt cannot restore untouched status. New attempts freeze their own candidate/config and report fresh_test or reused_test with prior release count. Local sealing is a workflow boundary, not an anti-cheating guarantee against a learner who can read local source files. Assessors see exposure history (capstone attempts) and feedback-opened-early and prediction-revision history (exercise records).

## INT-010 — Preserve complete import and reuse meaning
The archive extension is .lfbundle and the data manifest/schema is authoritative. Physical storage uses objects/<first-two-digest>/<digest>/<safe-name>, not archive entry paths. Exported IDs are historical; import returns fresh local IDs. With skip, an already imported identical source identity may satisfy a dependency through its existing local mapping; no new logical record is created for it. Missing/ambiguous dependencies reject the whole apply. The only exception is the pinned pretrained base omitted by a thin bundle: the import records it as base_required and stores the adapter's base_model_id as null; at use time the adapter binds to any prepared base whose prepared_manifest_sha256 equals the adapter's model_manifest_sha256, otherwise chat, evaluate or resume returns 400 VALIDATION_FAILED reason SUBJECT_INCOMPATIBLE. The duplicate key for skip is (original_installation_id, original_object_type, original_object_id, content sha256, and for checkpoints portability); an identical key maps to the existing local entity (disposition skipped_existing); the same original identity with different content or portability is retained as a distinct entity (disposition created or copied), exactly like divergent notes, and is never skipped. identity_map covers every imported entity type (artifact, dataset, run, model, checkpoint, evidence, assessment, note, conversation, tokenizer, retrieval_index) with a non-null local_id. Archive entries use the path llm-foundations-bundle/objects/<object_type>/<original_object_id>/<safe-name> with object_type from the closed bundle-manifest vocabulary; a full bundle carries exactly the eight pinned base files, a thin bundle none of them plus a thin_bundle omission. Only archive size is checked at upload (413 PAYLOAD_TOO_LARGE); archive-structure, digest and semantic failures fail the validate_bundle or import_bundle job with ARCHIVE_UNSAFE, CHECKSUM_MISMATCH or SEMANTIC_INVALID under the bundle semantic rules. With copy, fresh IDs and complete internal reference rewrites are mandatory.

Full versus thin bundles and include_resume_state are explicit choices. Inference-only adapter/tiny weights can generate, but tiny_resume or adapter_resume naming a checkpoint whose portability is not `resume` MUST be rejected before any job row with 400 VALIDATION_FAILED, reason_code CHECKPOINT_NOT_RESUMABLE, field_path /checkpoint_id. Complete resume state includes optimizer/RNG and immutable training configuration. No loader interprets an omitted optimizer as fresh training. Imported weights stay imported provenance even when a new local generation successfully reads them.

Validation reports safe structure/digests; import registers; actual load/evaluate/generate demonstrates reuse. These are three different observations. A valid thin bundle without its exact base is labelled base_required and cannot claim offline runnable. Full-bundle qualification uses an empty root and network disabled.

## INT-011 — Ship concrete teaching inputs and verifier profiles
All deterministic fixture texts, labels, split membership, ID formulas and hashes MUST be supplied in this package's fixture definitions/materialized data. Implementers MUST NOT invent the remaining facts, create labels with a model, download another corpus or replace measured outcomes with expected-looking curves. Illustrative arithmetic, actual learner measurements, supplied recorded runs and synthetic teaching text retain distinct labels.

Every evidence verification_profile_id MUST use the exact closed dispatch and role/rule mapping in [evidence-verification.md](contracts/evidence-verification.md), with fields/oracles in CUR and identities in DAT/APP. AC-CUR-020, AC-DAT-014 and AC-RUN-020 exercise these joins. Verifiers check objective artifacts and arithmetic, not semantic helpfulness or reasoning prose. The root document checker validates fixture/contract correspondence only; real workers and learner assessments remain product acceptance work.

## INT-012 — Resolve amendments before qualification
The implementation MUST fail its contract checks on drift between prose, OpenAPI, schemas, fixtures or case registry. A documented amendment increments spec revision, explains old/new behavior and updates every consumer and case. No ambiguity grants the builder permission to silently choose a dependency, model, threshold, rubric or format.

The independent document review and root resolution records are retained in reviews/. Product qualification uses DEL's full case/profile/browser denominator on one frozen candidate. Document checks, source preservation and published bytes do not establish native behavior, model quality or learning effectiveness.
