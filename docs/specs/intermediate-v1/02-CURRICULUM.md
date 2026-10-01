# Curriculum and assessment specification

Status: normative implementation specification. This document defines the intermediate-edition curriculum overlay and its evidence rules. It does not report implementation, browser testing, learner outcomes, or educational efficacy.

The preserved foundation course begins with a learner who can run Python and read a small function ([source](../../../course/START_HERE.md#L3)). It already teaches the fresh-model path, including data splitting, bounded training, evaluation, checkpoint reuse, and a controlled comparison. Its own text identifies the next gaps: exact-duplicate checks do not find every near duplicate or shared template ([lesson 03](../../../course/lessons/03-data.md#L7)); eight fixed evaluation batches are not a confidence interval ([lesson 08](../../../course/lessons/08-evaluation.md#L11)); and describing LoRA or placing a chat shell around a continuation model does not perform instruction tuning ([lesson 11](../../../course/lessons/11-modern-models.md#L23)). This edition turns those boundaries into executed exercises while retaining the original 00–12 lesson bodies.

Machine-readable curriculum constants are frozen in [curriculum-overlay.json](fixtures/curriculum/curriculum-overlay.json), worked-example oracles in [exercise-oracles.json](fixtures/curriculum/exercise-oracles.json), the capstone scoring instrument in [capstone-rubric.json](fixtures/curriculum/capstone-rubric.json), and the exact per-item verifier joins in [evidence-verification.md](contracts/evidence-verification.md).

## CUR-001 — Preserve the foundation and add a versioned overlay

The implementation MUST retain lesson IDs 00–12 and their canonical Markdown bodies byte-for-byte at the source-baseline identity in [00-DECISIONS.md](00-DECISIONS.md). The intermediate edition MUST be a separately versioned overlay containing P00, 13–20, E01, E02, supplement-card metadata, exercise metadata, and assessment metadata. It MUST NOT replace, renumber, paraphrase, or silently migrate foundation content.

The edition manifest MUST use `edition_id: "intermediate-v1"`, MUST identify the foundation edition and content digest, and MUST order modules as P00, 00–20, E01, E02. P00 and both electives MUST be shipped and reachable but MUST NOT count as required intermediate completion. Modules 13–20 MUST be required. Existing lesson 12 evidence MUST remain a prerequisite for module 20.

Every module MUST be readable in static and companion modes. A section requiring Python execution MUST show a `Companion required` label in static mode, the exact operation name, an explanation that no Python ran, and a link to local setup. Exact browser tokenization MUST use `Tokenizer calculation · exact browser result`. Structural views MUST use `Structural simulation · no model execution`; fixed pedagogical traces use `Illustrative trace`. Active companion observations MUST use `Local companion job` plus the explicit job state; queued and starting MUST NOT be described as executed. Supplied and imported measurements MUST use `Recorded result · supplied snapshot` and `Recorded result · imported snapshot`. Terminal local/bundled results MUST retain job or bundle provenance. A presentation-only state MUST NOT satisfy an artifact requirement.

## CUR-002 — Use explicit prerequisites, sessions, and completion states

The following graph and budgets are normative. A time is a reading/practice planning budget, not a runtime or graduation promise.

| ID | Title | Prerequisites | Planned sittings | Required |
|---|---|---|---:|---|
| P00 | Python, JSONL, and experiment refresher | Basic terminal access | 2 × 20 min | No |
| 13 | Data clinic | 03; 12 recommended | 3 × 25 min | Yes |
| 14 | Training diagnosis | 07, 09, 13 | 3 × 25 min plus run time | Yes |
| 15 | Evaluation and uncertainty | 08, 10, 14 | 3 × 25 min plus run time | Yes |
| 16 | Pretrained baseline | 11, 15 | 3 × 25 min plus explicit download/evaluation | Yes |
| 17 | Adapter fine-tuning | 16 | 4 × 25 min plus run time | Yes |
| 18 | Generation and conversation | 09, 16, 17 | 3 × 25 min | Yes |
| 19 | Packaging and reuse | 09, 18 | 3 × 25 min | Yes |
| 20 | Independent applied capstone | 12–19 | 4–6 sittings plus run time | Yes |
| E01 | Safe model modification: weight tying | 06, 09, 15 | 3 × 30 min plus tests/run time | No |
| E02 | Retrieval with evidence | 11, 16, 18 | 3 × 25 min | No |

Prerequisites MUST control practice launch and artifact verification, not reading access. curriculum-overlay.json splits each module's prerequisites into `blocking_prerequisites` (applied modules 13-20, plus lesson 12 for module 20, each satisfied only by that module's evidence - or FOUNDATION-12 - being artifact_verified in this registry) and `advisory_prerequisites` (foundation lessons, shown as recommended and never blocking, per SCP-002). Verifying evidence while a blocking prerequisite is unverified returns 409 STATE_CONFLICT reason PREREQUISITE_NOT_VERIFIED. A learner MAY read ahead. A launch blocked by a prerequisite MUST identify every missing prerequisite and the precise required evidence state.

The UI MUST keep `read`, `practiced`, `self_checked`, artifact verification, and review status separate. Progress v2 stores the first three as independent booleans. `read` requires voluntary `Mark read`; opening, scrolling, or time cannot set it. Only the server (or, in static mode, the sidecar logic) sets `practiced`, when it accepts an independent_submission exercise record, and `self_checked`, when a check record is followed by a feedback_opened record for the same exercise; PATCH /progress may set only `reading` and `read` to true and rejects practiced/self_checked true with 400 VALIDATION_FAILED reason PROGRESS_DERIVED_FLAG (semantic rule PROGRESS_DIRECT_FLAGS); setting any flag false requires confirm_reset. Evidence separately derives `verification_status` and `review_status`. No state implies certification, general model quality, job readiness, or measured efficacy.

Modules 13–19 reach `artifact_verified` only when their listed objective artifacts pass. Module 20 reaches `artifact_verified` only when the CUR-012 objective verifier passes; its separate assessment decision is derived by CUR-013. Assessment records use `review_kind:"self"` or `"independent"` with reviewer, rubric version, decision, rationale, and time. Self review derives `self_reviewed`, never an independent badge. Independent review requires a named reviewer and derives `independently_reviewed`.

## CUR-003 — Apply a fixed lesson grammar and fade scaffolding

Every new module MUST render these sections in this order:

1. **Outcome and boundary** — one observable outcome; named operations; what does and does not update weights.
2. **Predict before running** — a committed answer saved before results or feedback are revealed.
3. **Guided worked example** — fixed inputs, intermediate values, final answer, and why tempting alternatives fail.
4. **Partially scaffolded practice** — a fixture with some steps withheld; field-level feedback after submission.
5. **Independent evidence task** — a fresh submission using the module's artifact contract.
6. **Feedback and limits** — deterministic results separated from rubric-assessed reasoning.
7. **Stopping point** — named saved artifacts and the exact next action.

Scaffolding MUST fade without changing the assessment oracle:

| Band | Modules | Feedback before independent submission |
|---|---|---|
| Guided | P00, 13, 14 | Worked answer plus field-specific hints; learner must first save a prediction |
| Partial | 15, 16, 17 | Worked answer; at most one optional hint per withheld step; no prewritten conclusion |
| Sparse | 18, E01, E02 | Contract and error messages; no procedural sequence after prediction |
| Independent | 19, 20 | Acceptance fields and rubric only; no command sequence or recommended conclusion |

A prediction MUST be immutable after its associated operation starts; a learner MAY append a revision after seeing evidence. Reference answers MUST remain available for accessibility and self-study, but opening one before independent submission MUST be recorded as `feedback_opened_early: true` and MUST NOT prevent practice or artifact verification. It MUST be visible to a later assessor. Exercise interactions are append-only exercise records ([exercise-record.schema.json](contracts/schemas/exercise-record.schema.json)) created with POST /api/v1/exercise-records for kinds prediction, partial_practice, independent_submission, check and feedback_opened, each naming a compiled exercise_id (unknown IDs are 400 VALIDATION_FAILED reason EXERCISE_UNKNOWN). Records are never edited or deleted. The server derives `feedback_opened_early` (true when a feedback_opened record for the same exercise precedes the independent_submission) and `revised_after_job_start` (true when a prediction supersedes one whose linked job had left queued). Assessors read the history with GET /api/v1/exercise-records?module_id. Static mode stores the same records in its sidecar and derives the same flags client-side, labelled as browser-recorded.

## CUR-004 — P00 refreshes prerequisite mechanics without awarding intermediate evidence

P00 MUST cover: choosing the current Python interpreter; distinguishing Windows and WSL paths; reading and writing one JSON object per physical JSONL line; recognizing UTF-8 bytes versus characters; reading the shape `[batch, positions, width]`; computing a mean; and preserving an output directory instead of overwriting it.

Its prediction asks whether `[12,9,9,3]` produces inputs `[12,9,9]` and targets `[9,9,3]`; the answer is yes. Its guided JSONL example contains two valid records followed by `{"id":"R3","text":}`; line 3 is invalid JSON and MUST NOT be skipped. Its partial mean of `1.2, 1.4, 1.3` is `1.3` within `1e-12`. Its independent task runs startup preflight and saves only `storage_root_display`, interpreter label, Python patch version, release profile, and fixed write/read/delete outcome. It MUST NOT expose an absolute host path.

P00 MUST stop after producing `prerequisite_check_v1`. The check MUST be artifact-verified exactly when every P00 check in contracts/evidence-verification.md passes and otherwise MUST remain unverified; in both cases it MUST NOT contribute to required-module counts or any intermediate badge. Failure feedback MUST distinguish missing Python, wrong version, unavailable storage, and profile mismatch. Missing or wrong Python and profile mismatch are detected by the launcher before the service starts, so they are reported on stderr with fixed exit codes: 10 unsupported Python version (message names the found and required versions), 11 storage root missing, not a directory or not writable, 12 invalid OS/profile pair, 13 port or instance lock occupied; the P00 lesson explains each code in static mode. `storage_root_display` is `~/` followed by the root's path relative to the user's home directory when the root is inside it, otherwise `.../` followed by the root's final path component.

## CUR-005 — Module 13 performs a leakage-aware data clinic

Module 13 outcome: produce a dataset audit and a group-disjoint split without training a model. It MUST state that data auditing changes neither tokenizer rules nor neural weights.

The fixed `data-clinic-v1` fixture contains 48 synthetic support-note documents: 12 `scenario_group_id` values with four variants each. The clean split is groups DC01–DC06 in train (24 records), DC07–DC09 in validation (12), and DC10–DC12 in test (12). The companion MUST audit `data-clinic-leaky-v1`, an audit-only mutation containing exactly one cross-split exact-duplicate pair, two normalized near-duplicate pairs, and one group overlap. It MUST refuse to select the leaky fixture for training or evaluation.

The prediction asks whether a random record split is safe when four rewrites share one scenario; the accepted answer is no because related variants can cross splits. The guided example computes the leaky fixture's oracle: `exact_duplicate_pairs=1`, `normalized_near_duplicate_pairs=2`, `group_overlap_count=1`. The partial practice audits the clean fixture; accepted values are train/validation/test `24/12/12` and zero for all three leakage counts. Near-duplicate normalization and pair ordering MUST come from the data contract rather than prose inference.

The independent task MUST retain a `llm-foundations-data-audit-v1` artifact with dataset ID, eligibility, record/UTF-8/split counts, exact/normalized-near/group-overlap counts, algorithm version, and rejection strings; and a linked `llm-foundations-dataset-v1` manifest with split artifact IDs/hashes/counts/bytes and `provenance_note`. That note records source, permitted use, split method, removals, and limits. For the clean fixture, rejections are empty and the note acknowledges synthetic shared style. Deterministic fields are machine checked; the note is self/human assessed.

Module 13 reaches `artifact_verified` only when both audit oracles match, no leaky dataset was accepted for compute, and the linked manifest/note are present. It stops with clean/leaky audit IDs and a frozen clean dataset ID for module 14.

## CUR-006 — Module 14 diagnoses configuration, interruption, and unstable evidence

Module 14 outcome: distinguish a request rejection, a failed job, an interrupted job, and a completed run, then justify the next bounded action. It MUST preserve the distinction between `tiny_train`, `tiny_resume`, `evaluate`, and `generate`.

The prediction presents an illustrative trace with validation losses `3.20, 2.80, 3.10` at steps `50,100,200`; the accepted best measured checkpoint is step 100, not step 200. The worked example presents `width=63, heads=4`; the accepted diagnosis is a configuration error because width is not divisible by heads. No training job starts: the diagnostic request reports `CFG_HEAD_DIVISIBILITY` at `heads` and launches no worker or job. It then presents a run interrupted after update 73 with a last safe checkpoint at step 50; the accepted resume point is 50.

The partial practice uses the labelled `diagnosis-traces-v1` fixture. Exact oracle labels are: `CFG_HEAD_DIVISIBILITY` for case D14-01; `DATASET_LEAKAGE_BLOCK` for D14-02; `NON_FINITE_LOSS` for D14-03; `USER_CANCELLED_SAFE_BOUNDARY` for D14-04; and `NO_QUALITY_FAILURE` for D14-05, where a completed run has poor-looking samples but finite valid measurements. An illustrative trace MUST never enter the Runs registry as a recorded run.

The independent task MUST (a) retain the invalid response; (b) run clean `tiny_train` on `data-clinic-v1` with byte tokenizer, context/width 64, 4 heads, 2 layers, batch 8, learning rate 0.0003, seed 17, 50 updates, and evaluation every 25; (c) submit a separate run with the same settings plus the non-editable pair `exercise_profile_id:"tiny-v2-diagnosis-v1"` and `curriculum_hold_after_step:25`; (d) cancel at `phase_changed:cancellable_hold`; and (e) `tiny_resume` the step-25 checkpoint for 25 updates. Cancellation MUST yield `interrupted` at step 25. Ten minutes without action MUST yield `interrupted/exercise_timeout`, never continue. The UI labels a live teaching hold and exposes no general pause/hold control. No loss threshold applies.

The objective `training_diagnosis_v1` artifact MUST contain the request/error code, clean job identity and terminal state, cancellation job identity and terminal state, safe checkpoint step or explicit absence, resume parent/child lineage (the `tiny_resume` parent checkpoint ID and child run ID, which module 14 requires), unchanged-versus-changed settings, and a learner-authored diagnosis for each. Machine checks verify states, identities, step arithmetic, and lineage. The explanation of likely cause and proposed next experiment is self/human assessed.

Module 14 stops when the clean run and cancellation evidence are retained. The next action is to select checkpoints using validation only; test data MUST remain unused until module 15.

## CUR-007 — Module 15 measures uncertainty and performs error analysis

Module 15 outcome: report variation across controlled runs, reserve test evaluation until selection is frozen, and inspect the worst measured records without turning a point estimate into a guarantee.

The prediction uses illustrative test NLL-per-UTF-8-byte values `1.2, 1.4, 1.3`. The accepted mean is `1.3`; the sample standard deviation is `0.1`; the range is `0.2`, each with tolerance `1e-12`. These numbers are labelled `Illustrative trace`, not imported measurements. The worked example shows two checkpoints with validation values 1.50 and 1.40 and test values 1.10 and 1.60. Selection MUST use the validation value, so the second checkpoint remains selected; choosing after seeing test values is test leakage.

The partial practice receives five per-record values and MUST order them by descending `nll_per_utf8_byte`, breaking ties by ascending `record_id`. It selects exactly the first three. The aggregate MUST be calculated as `negative_log_likelihood_sum / utf8_bytes_sum`; it MUST NOT average the per-record ratios. The response MUST explain that NLL-per-byte permits a representation-normalized comparison within this protocol but does not establish broad language ability.

The independent task MUST create three fresh `tiny_train` runs over `data-clinic-v1` with the byte tokenizer, context/width 64, 4 heads, 2 layers, batch 8, learning rate 0.0003, 100 updates, evaluation every 25, and seeds 17, 23, and 31. For each run, the learner selects the run's `best_checkpoint_id` (lowest validation token NLL, earliest step on ties) before any test request and evaluates it with subject `{kind: tiny_checkpoint, checkpoint_id}`. The three frozen checkpoints are each evaluated once on the fixed test split using `tiny-nll-per-byte-v1`, producing metric payload `test_nll_bytes_v1`.

Each evaluation MUST record aggregate `negative_log_likelihood_sum`, `utf8_bytes_sum`, `record_count`, and `nll_per_utf8_byte`; and ordered per-record `subject_index` (0 for a single subject), `record_id`, `utf8_bytes`, `token_count`, `negative_log_likelihood`, and `nll_per_utf8_byte`. The learner MUST report the three values, arithmetic mean, sample standard deviation using denominator `n-1`, minimum, maximum, and range. No required loss or variation threshold exists.

The learner MUST classify the three worst teacher-forced NLL records per seed as `unseen_wording`, `number_or_identifier`, `long_context`, `punctuation_or_unicode`, or `other`, citing a source span and explanation. These are assessor judgments, not generated-output failures; selection and cited source are objective. `evaluation_report_v1` is artifact-verified only when seeds/settings match, selection predates test, dataset/protocol identities match, arithmetic recomputes within `1e-12`, and ordering is exact. Conclusions/categories remain rubric assessed.

Module 15 stops with the three selected checkpoint IDs, their evaluation artifact IDs, and one bounded claim plus one explicit non-claim. The next action is to establish an unchanged pretrained baseline before adapting it.

## CUR-008 — Module 16 establishes a frozen pretrained baseline

Module 16 outcome: prepare the allowlisted pretrained model, evaluate it without weight updates, and retain a baseline that later adapter results can be compared against. It MUST distinguish explicit model download (`model_prepare`), evaluation (`evaluate`), generation (`chat_generate` for the instruction model), and training.

Before download, the learner predicts whether `model_prepare` teaches the model the course task. The accepted answer is no: it downloads and validates selected pretrained files; it does not run an optimizer. The UI MUST show repository `HuggingFaceTB/SmolLM2-135M-Instruct`, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, declared license, selected files, expected byte sizes and hashes, destination, and network action before enabling confirmation. Declining MUST leave no prepared model registry entry.

The guided scoring example has expected intent `password_reset`. Candidate A is exactly two lines, `INTENT=password_reset` then `REPLY=Open Settings, then Security.`, and scores `exact_intent_match=1` and `response_schema_valid=1`. Candidate B is plain text `password_reset` and scores `0/0`. Valid output has exactly two nonempty lines; the second begins `REPLY=` and contains at most 240 Unicode scalars. The response's semantic quality is not graded by these metrics.

The partial practice MUST identify which state changes in three actions: preparing the model changes the local model registry; evaluating writes measurements but not model weights; generating writes a transcript but not model weights. Any answer claiming an optimizer step in those actions is incorrect.

The independent task MUST evaluate the prepared base on all 12 `applied-intents-v1` test records in ascending `record_id` order using `applied-intents-greedy-v1`: temperature 0, top-p 1, and at most 64 new tokens. It saves the prompt, generated text, parsed `INTENT`/`REPLY` or parse error, expected intent, exact-intent/schema results, model/revision, dataset/split hash, and decoding settings per record. It reports overall and six per-slice `exact_intent_match` and `response_schema_valid_rate`. No minimum score applies.

The objective `pretrained_baseline_v1` artifact verifies all 12 records were evaluated exactly once, fixed decoding was used, metrics recompute, and the prepared base file digests remain unchanged before and after evaluation. The learner's account of strengths, failures, and applicability is self/human assessed. Module 16 stops with the immutable base `model_id`, baseline evaluation ID, and digest set. The next action is an adapter run that names this base explicitly.

## CUR-009 — Module 17 trains and compares a LoRA adapter

Module 17 outcome: train only the prescribed adapter parameters, resume the adapter, select by validation, and compare it with the unchanged baseline under one test protocol. It MUST state that adapter training starts from a selected pretrained base, freezes base weights, and differs from both fresh tiny-model training and tokenizer training.

The prediction asks whether a successful adapter run may change the base-model file digests. The accepted answer is no. The worked example compares `base_trainable_parameters=0` during adapter training with a nonzero adapter trainable count. The accepted check is identity between the observed trainable names/count and the `smollm2-intents-lora-v1` profile; total parameter count alone is insufficient.

The partial practice receives a manifest in which `k_proj` is trainable. The accepted result is rejection because the fixed profile permits LoRA rank 8, alpha 16, dropout 0 only on `q_proj` and `v_proj`. It also receives an example in which prompt tokens contribute to supervised loss; the accepted result is rejection because the fixed applied-intents recipe masks non-assistant tokens according to the applied-model contract.

The independent task MUST use `applied-intents-v1`, base identity from module 16, profile `smollm2-intents-lora-v1`, and seed 17. `adapter_train` MUST run 40 optimizer updates with evaluation and save every 10 updates. `adapter_resume` MUST then load the step-40 adapter checkpoint and optimizer/RNG state and perform 20 additional updates, ending at step 60. With batch size 1, gradient accumulation 4, and 60 shuffled training records, the prescribed total corresponds to four passes; the manifest MUST record this relationship without relabelling each optimizer update as an epoch.

Checkpoint selection orders validation greatest exact-intent numerator, greatest schema-valid numerator, lowest finite supervised-response mean NLL, then earliest update; null/nonfinite NLL ranks below finite. Only after selection, the implementation evaluates module-16 base and adapter on the same 12 test examples using `applied-intents-greedy-v1`. It reports overall/per-slice metrics, paired record changes, elapsed/resource fields, failures, and step-40→60 lineage. Improvement is not required.

Device comparison. Module 17 also runs one training job on two devices. Before it, the learner saves two predictions as exercise `M17-DEVICE`: which device finishes the 40-update `adapter_train` sooner, and whether the two runs' per-update training losses will be identical. The accepted answer to the second is no: the CPU and the GPU add float32 values in different orders, so the values agree closely but cross-device equality is not claimed (APP-008). The first has no accepted answer; the learner compares the prediction with the measurement, and a GPU is not guaranteed to be faster for a model this small. With a CUDA environment installed (RUN-001), the learner submits the independent task's `adapter_train` request (40 updates, seed 17, `smollm2-intents-lora-v1`) once from the OS CPU environment and once from the OS CUDA environment against the same storage root; these runs are labelled `Local companion job`. Without a CUDA environment, the lesson shows the supplied snapshot `fixtures/applied/device-comparison-reference-v1.json`, labelled `Recorded result · supplied snapshot` with its reference machine, and only after the prediction record exists. The lesson's device comparison table has one column per run and these rows: device and profile; training elapsed seconds at update 40; peak resident memory; peak reserved CUDA memory (GPU run only, `Not applicable` for the CPU run); the per-update training losses for updates 1 to 40, each with its absolute difference, and the largest difference; validation exact-intent, schema-valid and mean NLL at updates 0, 10, 20, 30 and 40; and each run's selected update. Every value comes from the runs' metrics artifacts or the snapshot, with six decimals for losses. No speed, memory or equality threshold applies, and the learner's explanation of the differences is assessed prose. On the verification machine (Ryzen 9 9900X, GeForce RTX 5070) the 40 updates took 88.1 s on the CPU (12 threads) and 42.9 s on the GPU under Windows, and 89.1 s (10 threads) and 42.7 s under WSL. Per-update losses differed between CPU and GPU from the first update, by at most 5.5e-5, and between the Windows and WSL GPU runs by at most 7.5e-7. Validation exact-intent and schema-valid counts matched at every evaluation in all four runs, and two CPU runs with the same profile and thread count were bit-for-bit identical ([evidence](evidence/qualification-windows-2026-10-01.json), [evidence](evidence/qualification-wsl-cuda-2026-10-01.json)). These are recorded observations, not expected values. The snapshot has format `llm-foundations-device-comparison-v1`: `machine` (OS, CPU model, logical cores and the RUN-015 GPU query line), `request` (the canonical `adapter_train` request), and `runs`, one per device, each with runtime_profile, device, Torch version, the per-update and validation rows copied byte for byte from the source run's metrics artifact, the selected update, peak resident bytes, peak reserved CUDA bytes (null for the CPU run) and the source metrics artifact's SHA-256. The builder produces it on the CUDA reference machine (DEL-002), and a later revision seals it with its SHA-256; until then, only the live path exists and CUR-009 cannot qualify for a learner without a CUDA environment.

The objective `adapter_comparison_v1` artifact reaches `artifact_verified` only when base identity/digests match module 16, trainable parameter names match the fixed profile, step arithmetic and resume lineage are correct, checkpoint selection used validation only, test identities/settings match, and all reported metrics recompute. Explanations of why cases changed and whether adaptation was useful remain self/human assessed.

Module 17 stops with base, adapter, selected-checkpoint, and paired-evaluation IDs, plus the two device-comparison run IDs or the supplied snapshot ID. The next action is to inspect how plain continuation and role-formatted conversation send different contexts to different model families.

## CUR-010 — Module 18 distinguishes continuation from conversation

Module 18 outcome: run plain continuation and role-formatted chat against explicitly selected model identities; inspect serialization and truncation; and prove that chatting did not update weights.

The prediction asks whether sending ten chat turns performs ten optimizer updates. The accepted answer is no. The guided example contrasts `generate` on tiny-v2, which receives one prompt string and continues it, with `chat_generate` on the pretrained base or adapter, which serializes system/user/assistant roles using the pinned chat template. A chat-shaped display MUST NOT route a tiny-v2 checkpoint through `chat_generate` or call its output instruction following.

The partial practice uses `chat-no-truncation-v1` and `chat-drop-oldest-v1`. For the first, the accepted truncation result is no dropped turns. For the second, the input MUST exceed the effective 512-token teaching context until exactly the oldest complete user-plus-assistant pair is dropped. The system message and latest user message MUST remain. The implementation MUST show raw serialized text, token count before and after truncation, IDs of retained and dropped messages, available input budget after reserving 64 new tokens, model identity, template identity, temperature 0, and top-p 1.

The independent task MUST perform: one `generate` continuation from a selected tiny-v2 `checkpoint_id` (any tiny-v2 checkpoint registered in this storage root; the module-18 verifier requires one tiny-v2 `generate` output and does not require the module-15 best checkpoint); one `chat_generate` request to the unchanged base; the same request to the selected adapter; and both truncation fixtures against the adapter. It MUST capture output token IDs/text, stop reason, raw serialized input, truncation decision, and base/adapter digests before and after. It MUST also export one conversation as a learner-controlled dataset candidate, but the export MUST remain unregistered until the learner separately selects it for dataset import. Chat history MUST NOT become training data implicitly.

The objective `conversation_boundary_v1` artifact verifies correct operation/model pairing, both truncation oracles, unchanged weight digests, and absence of a registered training dataset caused solely by chatting. Output helpfulness and instruction-following quality are not machine graded. Module 18 stops with the continuation and chat transcript artifact IDs. The next action is to package a selected result with its complete identity and verify reuse from clean storage.

## CUR-011 — Module 19 packages, imports, and reuses an experiment

Module 19 outcome: create a portable bundle, validate it without mutation, import it into an empty companion-owned storage root, and reproduce one same-profile inference from the imported identities.

The prediction asks whether a checkpoint file alone is a portable experiment. The accepted answer is no: tokenizer/template, configuration, model/base identity, weights or permitted pinned dependency, evaluation protocol, lineage, manifests, and hashes are required. The guided example is limited to the bundle inventory and error contract; the independent procedure is not shown.

The independent task MUST select either a complete tiny-v2 experiment or the module-17 base-plus-adapter experiment. `export_bundle` MUST produce one immutable archive and detached descriptor. `validate_bundle` MUST be read-only and verify schema version, inventory, sizes, SHA-256 digests, lineage, safe relative paths, required artifacts, and absence of credentials. The learner MUST launch a second companion instance with a new empty storage root, call `import_bundle`, and receive new local registry IDs linked to original immutable IDs and bundle digest. Import MUST NOT retain absolute paths from the source root.

Before export, the learner MUST run one fixed inference with temperature 0 and retain input plus output token IDs. After import on the same release profile, the learner MUST run the same operation and settings. The output token IDs and stop reason MUST match exactly. Cross-profile byte-for-byte inference is outside this oracle. The source root MUST remain readable and unchanged.

A second import of the identical archive into the destination MUST use `duplicate_policy:"skip"`, report the duplicate source identities, create no logical entity for them, and duplicate no immutable bytes. A one-byte-tampered copy MUST fail validation and import without mutation. `portable_reuse_v1` verifies digests, validation, lineage, duplicate result, tamper rejection, clean-root evidence, and same-profile inference equality. Portability limitations are self/human assessed.

Module 19 stops with source and imported IDs, bundle digest, validation artifacts, and paired inference records. The next action is module 20; it provides the brief and rubric but no command sequence.

## CUR-012 — Module 20 is an independent applied capstone

Module 20 outcome: answer a fixed new task brief by choosing and executing a bounded approach, preserving a frozen baseline, selecting with validation evidence, evaluating the sealed test exactly once, and delivering a reusable evidence bundle. The lesson MUST provide the problem, constraints, schemas, and rubric; it MUST NOT provide a command sequence, recommended approach, prewritten conclusion, or result threshold.

The fixed brief is: “Given a synthetic support request, return exactly two lines, `INTENT=<expected-slice>` and `REPLY=<response of 1 through 240 Unicode scalar values>`; preserve the ten evidence items listed under the evidence checklist below.” The deterministic `capstone-support-v1` dataset is separate from `applied-intents-v1` and uses the same six-label/output schema. It contains 48 examples: six intent slices by four scenario groups of paraphrased requests (3, 2, 1 and 2 records), with two groups per slice in train (30 records), one in validation (6) and one in sealed test (12). Its train and validation splits are visible from the start; sealed test inputs and answer key are released only through the final evaluation action. Related scenarios MUST remain group-disjoint.

Before compute, the learner MUST save: task interpretation; prediction; why an adapter is appropriate; comparison rule; fixed settings; validation selection rule; maximum compute; failure conditions; and unsupported claims. The required candidate is an adapter against the frozen base; prompt-only and retrieval runs may be retained as exploration but cannot substitute.

The adapter MUST use `smollm2-capstone-lora-v1`: pinned base; q_proj/v_proj LoRA rank 8, alpha 16, dropout 0; AdamW learning rate 0.001; seed 29; batch 1; gradient accumulation 3; 20 updates; evaluation/save every 5. Thirty records contribute 60 presentations, exactly two passes. Selection orders greatest validation exact-intent numerator, greatest schema-valid numerator, lowest finite supervised-response mean NLL, then earliest update; null/nonfinite NLL ranks last.

The guided miniature gives steps 10 and 15 equal intent/schema numerators, with finite mean NLL 1.2 and 1.1. Step 15 wins on lower NLL. It reveals no capstone test data.

The learner MUST establish a frozen base validation result before the candidate. The selected candidate manifest and selection decision MUST be sealed before test release. One explicit release token MUST authorize exactly one atomic `evaluate` job whose `subjects` are ordered `[frozen baseline, selected candidate]`. The service MUST consume the token in the same transaction that accepts the evaluate job, before any sealed byte can be read, evaluate both subjects over the same 12 records/settings, and emit one paired artifact and delta. Failure, cancellation, or partial subject failure still consumes the token and MUST NOT permit another sealed evaluation in that attempt. The UI MUST require a final confirmation that names both subjects and this consequence. No test result may be used to alter the candidate.

After a failed or cancelled paired evaluation, the learner MAY start a new capstone attempt with a new attempt ID, plan, frozen subjects, and release token. The previous attempt remains immutable and linked. Dataset exposure history MUST increment; the new attempt and report MUST say `Previously released capstone test`, include `prior_release_count`, and MUST NOT describe the test as untouched. A successfully completed paired evaluation closes that attempt; rerunning to optimize against its result cannot replace it. These history rules permit recovery without hiding test reuse.

The learner MUST then export, validate, import into an empty storage root, and reuse a thin or full bundle as defined by the data contract, then carry the destination's reuse receipt back to the origin root (RUN-016) where the capstone is verified. The reproduced inference MUST use the prescribed capstone prompt, which is the user message of `capstone-support-v1` validation record `cs-account_access-06` serialized by the APP-004 chat template with add_generation_prompt true, with temperature 0 and MUST match output token IDs on the same profile. Neither retrieval nor E01 is required and neither may substitute for a missing required artifact.

The evidence checklist MUST contain all of the following immutable references:

1. Foundation lesson-12 evidence identity and its verification state.
2. Capstone brief, prediction, operation justification, and pre-test plan timestamps.
3. Dataset ID/hash; group-disjoint audit; provenance; permitted-use and limitation notes.
4. Prepared base identity/digests and frozen validation evaluation.
5. Candidate manifest; completed/interrupted/failed attempts; validation records; deterministic selection record.
6. Test-release token identity, sealed request digest, ordered subject identities, and the one terminal paired test evaluation, including failure/cancellation if that occurred.
7. Overall and per-slice exact-intent/schema-valid results, plus raw output and oracle for all 12 test records.
8. Three worst/failing records selected by the metric contract, with cited raw evidence and learner categories.
9. Export descriptor, bundle digest, read-only validation, clean-root import lineage, tamper rejection, and reuse inference pair.
10. An explanation of at most 20,000 Unicode scalar values (the claim_text bound) of what changed, what did not change, what the evidence supports, what it does not establish, and one next experiment.

Objective gates are binary and MUST NOT grade prose quality: all required modules 13–19 and foundation lesson-12 evidence are present; all 10 checklist items have valid identities; the dataset audit passes; base digests stay fixed; adapter selection predates test release; release/attempt rules hold; metrics recompute; profile/operations match; bundle checks pass; and same-profile reuse matches. Passing sets only `artifact_verified` and `intermediate_evidence_packet_complete:true`.

## CUR-013 — Apply an exact capstone rubric without automated competency claims

The capstone explanation MUST be assessed with the 100-point instrument in [capstone-rubric.json](fixtures/curriculum/capstone-rubric.json). Ten rows are each scored only 0, 5, or 10; the table abbreviates the fixture, whose row titles and anchor texts are authoritative:

| Row | 0 | 5 | 10 |
|---|---|---|---|
| Problem framing | Missing/contradictory | Goal stated, operational choice weakly linked | Goal, operation, metric, and failure condition form one testable question |
| Data custody | Source/split absent | Identity present; leakage or permission limits incomplete | Identity, permission, grouping, leakage checks, and limits are linked to artifacts |
| Baseline | Missing or changed silently | Baseline present but comparison controls incomplete | Frozen identity, settings, outputs, and digest evidence support direct comparison |
| Candidate control | Multiple unexplained changes | Adapter named; one control or lineage gap | Adapter change is isolated and full lineage/settings are accounted for |
| Training and operation distinction | Confuses tokenizer, weights, resume, or inference | Correct labels with incomplete state explanation | Correctly explains every used operation and which persisted state it changes |
| Evaluation discipline | Test influences selection or claim exceeds metric | Selection is valid; uncertainty/error analysis thin | Validation selection, sealed test, slice results, failures, and uncertainty bound the claim |
| Diagnosis | Missing or invented cause | Observed failure cited; next action generic | Evidence separates symptom/cause hypotheses and proposes a discriminating bounded test |
| Reproducibility | Artifacts missing/unusable | Main run reproducible; environment or settings gap | Another qualified user can identify data, model, settings, lineage, metrics, and bundle |
| Reuse and safety | Bundle unchecked or source overwritten | Bundle validates; reuse/tamper evidence incomplete | Clean import/reuse, duplicate policy, tamper rejection, and source preservation are demonstrated |
| Claim quality | Makes broad quality/efficacy claim | Includes a limitation but unsupported positive wording | Every positive claim cites evidence and explicit non-claims cover data/model/evaluation scope |

The ten `rubric_rows` MUST occur exactly once and in this order: `problem_framing`, `data_custody`, `baseline`, `candidate_control`, `operation_distinction`, `evaluation_discipline`, `diagnosis`, `reproducibility`, `reuse_safety`, `claim_quality`. Each row stores that exact criterion ID, a score of 0, 5, or 10, and a nonempty comment. The server, not the assessor client, computes `total_score` as the integer sum of the ten row scores and computes `objective_gates_verified` from the CUR-012 verifier result. A client cannot submit or override either field.

The server derives the only valid decision as follows: `meets` if and only if `objective_gates_verified` is true, `total_score >= 80`, and no row is zero; otherwise `partially_meets` if and only if `total_score >= 50`; otherwise `does_not_meet`. Thus a score of 80 or more with a failed objective gate or zero row is `partially_meets`, and every score below 50 is `does_not_meet`. A submitted decision that differs from this derivation MUST be rejected as `SEMANTIC_INVALID` without storing an assessment.

Each stored assessment contains the immutable rows, computed total and gates, derived decision, reviewer, rationale, origin, and time. `review_kind:"self"` derives `self_reviewed` and label `Self-reviewed`; `review_kind:"independent"` with a named reviewer derives `independently_reviewed` and label `Independently reviewed`. Review status identifies who reviewed the work and does not imply `meets`. Imports promote neither. Neither review kind is certification or a measured intermediate-user claim. Later reviews append records.

Module 20 stops with immutable evidence export and a choice to request independent review or retain self review. It has no automatic graduation or quality claim.

## CUR-014 — E01 implements and verifies safe weight tying

E01 outcome: make one scoped architecture change in a learner-owned tiny-v2 workspace, prove its tensor/checkpoint consequences, and run it without changing the preserved foundation lab.

The prediction uses vocabulary 257 and width 64. Tying the token embedding and bias-free output projection removes one independently stored `257 × 64` matrix, so the accepted parameter delta is `16,448`. Width, vocabulary logits, and context do not change. The worked example MUST show the one shared parameter object, not two numerically equal copies.

The editable task MUST create a learner-owned tiny-v2 workspace outside `course/`. Only the region between the single `E01_EDIT_BEGIN` and `E01_EDIT_END` markers in `tiny_v2/model.py` may differ from the shipped workspace template. The learner adds weight aliasing so the output projection's `weight` is the same parameter object as the token embedding `weight`. Copying values, registering two independent optimizer parameters, tying position embeddings, changing vocabulary, editing outside the markers, or editing `course/labs/model.py` MUST fail verification.

Before training, the learner MUST run `python -m llm_foundations_companion verify-e01-workspace --workspace <absolute-dir> --output <unused-output-directory>`. Fixed inputs are seed 17, vocabulary 257, context 16, width 64, heads 4, layers 2; forward rows 0–15 and 16–31; and causal inputs `[4,6,2,9]` versus `[4,6,2,8]`. Checks are `E01-T01-SHARED-IDENTITY`, `E01-T02-PARAM-DELTA`, `E01-T03-FORWARD-SHAPE-FINITE`, `T04-CAUSAL-INVARIANCE`, `T05-SAVE-LOAD`, and `T06-CHECKPOINT-COMPAT`. The directory receives the hash-bound JSON receipt and fixed logs. Exit codes are 0 pass, 2 invocation/path/output-exists, 3 marker/protected-diff, 4 test failure, and 5 internal error.

The learner invokes the harness in a terminal; the browser MUST NOT upload or execute edited source. After a passing receipt, companion `tiny_train` references that receipt and runs shipped `architecture_profile_id:"tiny-v2-weight-tied-v1"`, mapped to `architecture_variant:"tiny-v2-tied-v1"`, for 50 updates, seed 17, `data-clinic-v1`, and evaluation every 25. The patch itself is tested only by the CLI; arbitrary learner code never enters the web job. Evidence links receipt and builtin profile. It retains diff/receipt, profile hash, manifest, metrics, checkpoint, and result. No loss threshold, superiority, or general learning claim is allowed.

E01 reaches `artifact_verified` only when the protected source is unchanged, the diff is confined to the learner workspace, all six checks pass, the allowed tied profile ran, and the comparison identities are complete. This awards only the `Safe model modification — weight tying` elective evidence badge.

## CUR-015 — E02 builds lexical retrieval with evidence

E02 outcome: build a deterministic lexical index, inspect ranked evidence, and cite the passage that supports a bounded answer. It MUST state that `retrieval_build` creates an index, not neural weights; `retrieval_query` ranks documents, not truth; and any later `chat_generate` call performs inference without updating either model or index.

The prediction asks whether a weekly manual is best refreshed by retraining adapters. The answer is no here: rebuild retrieval from the current authorized manual and test support. The worked example applies NFKC, casefold, ASCII tokens `[a-z0-9]+`; `tf=count/total_document_terms`; `idf=ln((1+N)/(1+df))+1`; L2 normalization; float64 cosine; and descending score then ascending `record_id`. `top_k` accepts 1–20 and returns `min(top_k,N)` rows, including deterministic zero-score ties. A no-vocabulary query fails.

The deterministic `retrieval-manual-v1` fixture contains documents RM01–RM08 and queries RQ01–RQ06. With `k=3`, the complete ranking/support oracle is:

| Query | Ranked document IDs | Required supporting citation |
|---|---|---|
| RQ01 | RM01, RM03, RM07 | RM01 |
| RQ02 | RM07, RM02, RM04 | RM02 |
| RQ03 | RM03, RM01, RM08 | RM03 |
| RQ04 | RM04, RM07, RM08 | RM04 |
| RQ05 | RM08, RM05, RM07 | RM05 |
| RQ06 | RM06, RM07, RM02 | RM06 |

In RQ02 and RQ05 the highest-ranked document is a distractor that shares the query's words, and the supporting document is ranked second. The partial practice runs RQ01 and RQ02. It MUST require the learner to predict the top document before reveal, inspect score contributions, and distinguish “retrieved highest” from “supports the answer.” The independent task MUST run `retrieval_build` once and all six `retrieval_query` operations with the immutable index, query, tokenization, and ranking settings. For each query, the learner MUST write one answer of 1 through 20,000 Unicode scalar values in `claim_text`, cite one returned document, and quote or point to the exact supporting span. The machine oracle verifies index/dataset identity, top-three order, citation membership, and required supporting document. It MUST NOT grade answer fluency or factual completeness beyond the fixed support oracle.

The objective `retrieval_evidence_v1` artifact MUST contain index ID/hash, source document hashes, tokenizer/ranking contract identity, all query/ranking/score outputs, cited document/span identities, and build/query job states. Rebuilding after a source change MUST create a new index identity; earlier query evidence MUST continue to reference the old index. E02 stops with the artifact and awards only `Retrieval with evidence` elective evidence after objective checks. It cannot substitute for modules 13–20.

## CUR-016 — Insert exact supplement cards without changing lessons 00–12

The overlay MUST contain the following cards. `anchor_heading` identifies a section; the card renders after that section's final block and before the next heading. `anchor_occurrence` is 1 for every row. At the same anchor, ascending `order` controls card order. Routes derive as `#/learn/{destination_module_id}`.

| card_id | source | anchor_heading | order | kind / visible label | destination | title | summary | minutes | prerequisites |
|---|---|---|---:|---|---|---|---|---:|---|
| SC-00-P00 | 00 | `## Try` | 10 | prerequisite / Optional refresher | P00 | Refresh Python and experiment basics | Check JSONL, shapes, arithmetic, and your selected environment. | 40 | — |
| SC-03-13 | 03 | `## Check` | 10 | required / Required practice | 13 | Audit related data before splitting | Detect exact, near, and scenario-group leakage in fixed fixtures. | 75 | 03 |
| SC-07-14 | 07 | `## Check` | 10 | required / Required practice | 14 | Diagnose runs without erasing evidence | Separate rejected, failed, interrupted, and completed work. | 75 | 07,09,13 |
| SC-08-15 | 08 | `## Check` | 10 | required / Required practice | 15 | Measure variation and hold test data back | Compare three seeds and perform record-level error analysis. | 75 | 08,10,14 |
| SC-11-16 | 11 | `## Choose the operation` | 10 | required / Required practice | 16 | Establish a pretrained baseline | Prepare the pinned model and measure it without training. | 75 | 11,15 |
| SC-11-17 | 11 | `## Choose the operation` | 20 | required / Required practice | 17 | Train only an adapter | Resume LoRA and compare it with the unchanged base. | 100 | 16 |
| SC-09-18 | 09 | `## Check` | 10 | required / Required practice | 18 | Inspect continuation and chat context | Compare plain prompts, role serialization, and truncation. | 75 | 09,16,17 |
| SC-09-19 | 09 | `## Check` | 20 | required / Required practice | 19 | Package and reuse the exact result | Validate, import, and reproduce inference from clean storage. | 75 | 09,18 |
| SC-12-20 | 12 | `## Finish` | 10 | required / Required practice | 20 | Complete an independent applied experiment | Choose a path, seal the test, and deliver linked evidence. | 100 | 12–19 |
| SC-06-E01 | 06 | `## Check` | 10 | elective / Elective | E01 | Safely tie model weights | Make one architecture change and prove its consequences. | 90 | 06,09,15 |
| SC-11-E02 | 11 | `## Work a decision` | 10 | elective / Elective | E02 | Retrieve and cite current evidence | Build deterministic TF-IDF retrieval and inspect support. | 75 | 11,16,18 |

Each card object MUST contain `card_id`, `source_lesson_id`, `anchor_heading`, `anchor_occurrence`, `order`, `kind`, `destination_module_id`, `title`, `summary`, `estimated_minutes`, and `prerequisites`. Build validation MUST fail if the source lesson is absent; the exact heading occurrence is absent; another heading produces ambiguity; a card/destination ID is duplicate; order duplicates at one anchor; kind is invalid; a prerequisite ID is unknown; or the destination route is absent. There is no fallback to the top, bottom, or a fuzzy heading match.

Foundation lesson rendering and voluntary progress remain unchanged. A card click MUST NOT mark the source read, mark the destination practiced, or satisfy a prerequisite.

## CUR-017 — Keep deterministic feedback separate from assessed reasoning

Names of the form `training_diagnosis_v1`, `evaluation_report_v1`, `pretrained_baseline_v1`, `adapter_comparison_v1`, `conversation_boundary_v1`, `portable_reuse_v1`, `retrieval_evidence_v1` and `prerequisite_check_v1` in this document denote a module's evidence submission as checked by its verification profile; they are not artifact types. Learner-authored diagnoses, categories, cited spans and reported statistics live in claim_text, note_refs or structured_response; every statistic the verifier checks is computed by the verifier from registered artifacts and reported as a received_value in verification_checks.

The implementation MUST evaluate objective fields from versioned fixtures and schemas. String identifiers and categorical answers use exact equality; unordered sets are normalized only when the contract explicitly declares a set; numeric tolerances are declared per oracle; hashes use exact lowercase SHA-256; record ordering uses the declared tie-break. A changed fixture version MUST create a new oracle identity and MUST NOT silently regrade stored submissions.

Objective feedback MUST name the failed field, expected rule, received value, and source artifact. It MUST NOT reveal an independent-task answer before submission. Runtime or infrastructure failure MUST yield `NOT_RUN` for dependent checks, not zero points or a fabricated incorrect answer. A rerun appends an attempt with its own identities and timestamps.

Machine verification MUST NOT score provenance prose, permitted-use judgment, diagnoses, causal explanations, claim boundaries, error categories, or next-experiment arguments. The interface distinguishes objective checks, self review, independent review, and unassessed fields.

The course MUST NOT claim that completion caused learning, that a model became generally useful, or that the learner is professionally intermediate. Allowed statements report practiced modules, verified artifacts, self review, or named independent review.
