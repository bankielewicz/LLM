# Round 02 independent cross-section review

Review date: 2026-09-28 America/New_York  
Reviewer role: independent specification reviewer  
Scope: current `02-CURRICULUM.md` through `06-APPLIED-MODELS.md`, `contracts/**`, `fixtures/**`, and section trace registries. Root documents were reviewed separately in round 01.

This was a read-only specification review until this authorized review record was written. No application code, dependency installation, model download, model execution, browser journey, native profile qualification, or product acceptance case ran. Every product case remains `NOT_RUN`.

## Findings

### R02-01 — HIGH — E01 verification has no complete registration and execution path

`02-CURRICULUM.md:223-235` requires a learner-owned source change, the fixed CLI harness, a 50-update run, and a hash-bound receipt. `06-APPLIED-MODELS.md:144-154` defines `verify-e01-workspace` and says a failed receipt cannot register the tied profile. `04-RUNTIME-AND-API.md:13-20` omits that CLI entrypoint, while lines 40-42 and 62 forbid worker paths, uploaded code, and dynamic modules. `contracts/openapi.json` `TinyTrainRequest` accepts `architecture_profile_id: "tiny-v2-weight-tied-v1"` without a verification-receipt identifier.

The tied job can therefore bypass the learner-source verification, and the specification does not identify which process executes the learner source.

Exact correction: add the CLI to RUN-001/RUN-002 and freeze one execution model. The CLI must run learner source only in a bounded child, with exact CPU, time, memory, network, and writable-root limits, then register an immutable `e01_verification_id` bound to source tree, diff, harness, dependency lock, and profile. A tied `tiny_train` request must require that identifier. State explicitly whether the 50-update evidence executes the learner implementation or a companion-owned implementation; the product must not attribute the latter to learner code.

### R02-02 — HIGH — Capstone rubric rows cannot be submitted or retained

`02-CURRICULUM.md:202-219` requires ten named rubric rows, scores restricted to 0, 5, or 10, comments, an arithmetic total, and the published `>=80 and no zero` decision. `contracts/openapi.json` `AssessmentRequest` and `Assessment` contain only review kind, reviewer, rubric version, overall decision, and rationale. `contracts/schemas/assessment-record.schema.json` has the same omission.

An assessor can assert `meets` without the row evidence needed to recompute the result.

Exact correction: add rubric identity/version and exactly ten `{row_id, score, comment}` records for the capstone rubric. Derive total and decision server-side and reject a client decision inconsistent with the frozen rule. Persist attempts immutably and add boundary cases for a total of 80, a total of 85 with one zero, an unknown row, duplicate row, and invalid score.

### R02-03 — HIGH — Sealed capstone state is not representable and token types conflict

`02-CURRICULUM.md:176-200` and `05-DATA-AND-ARTIFACTS.md:132` require a frozen baseline, candidate, dataset/configuration, prior-attempt linkage, release history, consumed state, and terminal paired evaluation. OpenAPI `CapstoneAttempt` contains only `attempt_id`, `release_token`, `token_state: "available"`, a fresh/reused flag, and time. A GET cannot represent consumption, terminal job, frozen subjects, or lineage.

`contracts/schemas/sealed-evaluation-token.schema.json` requires a 43-character base64url token, while OpenAPI `EvaluateRequest.release_token` and `CapstoneAttempt.release_token` use UUID identifiers. The data token uses split `sealed_test`, while OpenAPI forces a release-token evaluation to split `test`.

Exact correction: use one 256-bit token representation and one split name throughout. Return the secret once at attempt creation. Persist and expose the complete freeze plus `available|consumed`, consumed time, job/evaluation identity, prior attempt, prior release count, and exposure label, without returning a reusable secret after consumption.

### R02-04 — HIGH — Prompt-only capstone candidates cannot be represented

`02-CURRICULUM.md:176-181` permits a prompt-only path that compares two supplied prompt templates and freezes the selected prompt before sealed evaluation. OpenAPI capstone subjects and evaluate subjects carry only model or adapter identity. `06-APPLIED-MODELS.md:60-70` defines one fixed applied evaluation serialization and no prompt-template variant.

Both prompt candidates would therefore be the same base-model subject, and neither the attempt freeze nor paired evaluation can identify the selected prompt.

Exact correction: define a closed applied-evaluation subject containing model subject plus `prompt_template_id`, freeze the two allowed capstone prompt templates and their exact bytes/digests, include the template in attempt creation and paired evaluation, and retain it in every record and metric artifact. The baseline and candidate must remain distinguishable even when both use the same base weights.

### R02-05 — HIGH — The required paired comparisons conflict with the runtime operation contract

Module 17 and APP-011 require one two-subject base/adapter evaluation without a sealed-test token (`02-CURRICULUM.md:136-140`; `06-APPLIED-MODELS.md:116-122`). `04-RUNTIME-AND-API.md:164` says an evaluate request normally has one subject and has exactly two only when `release_token` is present. OpenAPI permits one or two subjects without enforcing that prose rule.

Exact correction: define two distinct paired modes. A nonsealed comparison may use exactly two compatible subjects on the ordinary applied test split without a release token. A sealed capstone comparison requires exactly two ordered subjects, the consumed token, and the sealed split. Freeze compatibility, ordering, and result discriminators for both modes in prose and OpenAPI.

### R02-06 — HIGH — The capstone adapter path has no executable recipe

`02-CURRICULUM.md:176-181` allows adapter training on `capstone-support-v1`, whose train split has 30 records. OpenAPI `AdapterTrainRequest` accepts only preset `smollm2-intents-lora-v1`. `06-APPLIED-MODELS.md:72-88` hardcodes the 60-record `applied-intents-v1` order, `randperm(60)`, four-record accumulation, and lesson-17 cadence.

Using that contract on the 30-record capstone split is undefined, including pass boundaries, update count, validation cadence, selection, and resume cursor.

Exact correction: add a capstone adapter preset with exact update limit, seed, batching, validation/save cadence, and selection rule, or generalize the common recipe to a frozen `train_record_count` derived from the registered dataset. In either case, replace hardcoded `randperm(60)` with the exact profile rule and add a native capstone train/resume case.

### R02-07 — HIGH — Chat history cannot persist a completed assistant turn or become the specified dataset candidate

`03-LEARNER-EXPERIENCE.md:95-106` requires bound chat history. `02-CURRICULUM.md:144-154` requires exporting one conversation as a learner-controlled dataset candidate. `05-DATA-AND-ARTIFACTS.md:74` requires explicit export followed by dataset import. `06-APPLIED-MODELS.md:22` says `chat_generate` does not update a conversation.

OpenAPI optionally binds `chat_generate` to a conversation ID/revision, but the result returns no conversation revision. `/conversations/{id}/turns` accepts only user turns. The OpenAPI turn links `generation_id`, while `conversation.schema.json` links `generation_artifact_id`. No endpoint creates a dataset-candidate artifact.

Exact correction: define either an explicit bound-generation action that commits server-derived user and assistant turns on successful completion, or a separate commit-generation endpoint that accepts a generation artifact and expected revision. Freeze failed, interrupted, and revision-conflict behavior. Use one generation-artifact identifier field. Add an explicit conversation export format and a separate ordinary dataset registration/audit action; export alone must not register training data.

### R02-08 — HIGH — P00 preflight has no API or receipt

`02-CURRICULUM.md:64-70` and `trace/CUR.json` AC-CUR-004 require interpreter, Python version, profile, working/storage context, and a write/read/delete check saved as `prerequisite_check_v1`. Runtime discovery returns runtime version, profile/device, storage display, and writable status, but no Python identity or storage-test receipt (`04-RUNTIME-AND-API.md:24`). OpenAPI has no preflight action.

Exact correction: perform a bounded companion-root write/read/delete probe at startup or through a fixed authenticated preflight endpoint. Return a hash-bound receipt containing a non-path interpreter label, Python patch version, profile/device, storage display, check result, time, runtime version, and instance identity. P00 must consume that receipt and keep it outside required completion credit.

### R02-09 — HIGH — Thin/full bundle and resume-state choices are absent from the API and manifest

`06-APPLIED-MODELS.md:126-132` requires explicit `thin` and `full` applied bundles and includes optimizer state only when a learner explicitly selects resume state. OpenAPI `ExportBundleRequest` contains only arrays of run, model, checkpoint, evidence, and note IDs. It has no bundle mode, conversation selection, or resume-state choice. Because adapter inference itself uses a checkpoint ID, selecting the adapter checkpoint ambiguously selects optimizer state. `bundle-manifest.schema.json` has no mode, omissions, dependency map, lineage map, or `contains_resume_state` field.

Exact correction: add `bundle_mode`, selected conversations, and per-checkpoint `{checkpoint_id, include_resume_state}`. Define an inference-only adapter payload/descriptor when resume state is false. Expand the manifest with mode, omissions, dependency/lineage map, base requirement, and resume-state flag. Freeze thin-with-base, thin-without-base, full-offline, and resume/no-resume cases.

### R02-10 — HIGH — Required deterministic corpora are not authored

`fixtures/data/canonical-fixtures.json:6-11` contains counts, ID patterns, and placeholders such as “learner-authored scenario fact table” and “fixed user message from authored slice template.” It does not contain the 48 data-clinic texts, 84 applied records, 48 capstone records and labels, or retrieval document/query text. `05-DATA-AND-ARTIFACTS.md:130` says release packaging must materialize the complete authored fact/template tables and forbids the implementation from inventing missing text.

Training inputs, split hashes, exact expected responses, and retrieval rankings therefore cannot be reproduced from the specification.

Exact correction: add the complete source fact/template tables or final JSONL fixtures with every record ID, group, split, slice, message, and canonical two-line expected response, plus expected file SHA-256 values. Sealing must be enforced by runtime access control; it must not depend on omitting the authoritative bytes from the specification.

### R02-11 — HIGH — Artifact schemas contradict the claimed closed contract and reject valid generation semantics

`05-DATA-AND-ARTIFACTS.md:120-130` says unknown properties are rejected except versioned metric payloads. The current schemas leave `run-manifest.config`, `checkpoint.model_identity`, checkpoint dataset bindings, metric payloads, and other nested objects open. `run-manifest.schema.json` also requires at least one dataset binding and a seed for every listed operation, including free-form `generate` and `chat_generate`, which have no dataset and must not fabricate one.

Exact correction: replace open objects with operation/backend-discriminated closed schemas. Require dataset bindings only for operations that consume datasets; make inapplicable fields absent. Bind tiny/applied identities explicitly. Metric payloads need a protocol/version discriminator and a closed schema per supported protocol. Add one-extra-field and missing/inapplicable-field rejection fixtures at each nested boundary.

### R02-12 — MEDIUM — Acceptance traces omit required profiles or assign impossible static behavior

`06-APPLIED-MODELS.md:160` requires applied acceptance on every named profile, and its CUDA resume rule is normative. `trace/APP.json` AC-APP-006, AC-APP-007, and AC-APP-012 omit `wsl-cuda`. The curriculum traces AC-CUR-002 and AC-CUR-013 list `static` while requiring local artifact attachment and rubric assessment even though companion-owned v2 evidence and assessments are unavailable in static mode.

Exact correction: add separate CUDA deterministic train/resume and bundle cases, or list `wsl-cuda` on the existing cases. Split static cases into read-only rendering/absence behavior and companion cases into mutations. A static case must not require local artifact verification or assessment submission.

### R02-13 — MEDIUM — Module identity bounds drift

The edition contains 24 module keys: P00, 00-20, E01, and E02. `04-RUNTIME-AND-API.md:96` says 25. OpenAPI note create/update patterns accept nonexistent modules 21-29, while `progress-v2.schema.json` correctly caps 24 and restricts the exact set. `notebook-entry.schema.json` does not restrict module IDs at all.

Exact correction: define one shared module-ID schema/enum and reference it from progress, notes, evidence, routes, imports, and generated content. Set the root limit to 24 and add rejection cases for 21, 29, E00, and unknown strings.

## Positive cross-section observations

- The current curriculum, data, applied-model prose, and metric oracle now agree on the two-line `INTENT=<slice>` / `REPLY=<text>` output and keep semantic helpfulness outside automatic grading.
- Tokenizer training, fresh neural training, resume, evaluation, generation, pretrained preparation, adapter training, continuation, and chat remain explicitly distinct.
- Static mode consistently avoids claiming that browser controls ran Python; local behavior is tied to an authenticated loopback companion.
- The capstone permits negative results and separates objective artifact verification from self or independent rubric review.

These observations are document-level consistency findings only. They do not establish implementation, native behavior, model compatibility, accessibility, installation, or learner outcomes.
