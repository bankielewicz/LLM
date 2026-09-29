# Scope, baseline, and release contract

This is a target implementation specification. Existing behavior is identified below; every SCP requirement defines work to implement and verify. Reading this document is not evidence that a service, model workflow, or assessment exists.

## SCP-001 — Preserve and identify the foundation
The build MUST start from source commit `3a47ea48da53cb9de3ff4727ca5f0f0b6f2b9bf8`. The immutable foundation edition is `2026-09-26`. All 43 files listed in `qa/source-hashes.json`, both retained `course/` and `dist/course/` copies, `dist/content.json`, and `dist/course.zip` MUST retain their baseline bytes. [source-baseline.json](evidence/source-baseline.json) records all 107 pre-existing tracked file hashes for this specification task. Future interface work may change frontend files but MUST NOT change these protected content/lab artifacts.
The source receipt itself MUST match SHA-256 60c3ecb36c24512d6b390b94ae504dffba244f1d41f1ce7367afb6ab00126bd8 at base tree f365fbefcea6dcf65dbb970b9640f5073154cfde. A replacement receipt cannot redefine the protected baseline.
The existing Python command-line labs remain separately runnable. New backend code MUST live outside `course/`; existing tests are preserved as regression inputs, not replaced by weaker assertions. A foundation correction requires a separately identified edition and migration amendment. It cannot be folded into this expansion.
Evidence: baseline `README.md:3-24`, `course/design/INTERACTION_CONTRACT.md:1-9`, and `course/labs/WORKBOOK.md:1-67`.

## SCP-002 — Define the learner and observable outcome
The required entry knowledge is running a Python command, recognizing variables/lists/functions, and using a terminal working directory. P00 supplies a diagnostic and refresher for this knowledge. A learner can read any lesson without meeting prerequisites; executing advanced labs can be blocked by concrete missing artifacts or runtime capabilities, never by a paid account or arbitrary completion checkbox.
Intermediate completion means the learner has supplied the required objective evidence and identified assessment of reasoning in [02-CURRICULUM.md](02-CURRICULUM.md). It MUST NOT mean a certification, a measured learning-effectiveness claim, a guaranteed useful chatbot, a loss threshold, or elapsed study time. The learner must demonstrate dataset preparation, bounded training/adaptation, diagnosis, comparable evaluation, checkpoint reuse, and a justified conclusion on a new task.
A self-reviewed submission and an independently reviewed submission MUST have different labels. Software MUST NOT infer conceptual understanding from navigation, a finished process, or an automatically valid artifact.

## SCP-003 — Ship a finite content expansion
The release MUST include:
- Unmodified foundation lessons 00-12, with separate supplement cards.
- Applied modules 13-20, exactly as sequenced in the curriculum contract.
- P00, E01, and E02 as shipped learner electives. They remain implementation requirements even when not required for the core learner completion badge.
- Worked examples, concrete exercises, explanation feedback, stopping points, and objective/human assessment boundaries for every new module.
- Deterministic teaching data and deliberate failure/leakage fixtures defined by the curriculum and data contracts.
No empty route, placeholder lesson, future-tense feature card, or link to an unwritten external tutorial may satisfy one of these deliverables. External sources supplement the authored exercise; they do not substitute for its procedure or scoring oracle.

## SCP-004 — Support two honest product modes
`static` mode serves lessons, in-browser tokenizer calculations, structural simulations, recorded-run import, notes, and progress export/import. It MUST display every imported/bundled result as recorded evidence. It MUST NOT expose enabled training, generation-from-checkpoint, chat-inference, or filesystem registration controls.
`local` mode serves the reader and a companion API from the same `http://127.0.0.1:8765` origin. Runtime capability is established by an authenticated handshake, not hostname alone. Real execution controls require an authenticated compatible service, selected artifacts, and successful preflight.
A public/hosted Site MUST NOT probe localhost, request private-network access, proxy job requests, or imply its controls execute the local Python lab. The learner moves progress between origins through the explicit export/import flow.
The static edition remains useful when model files or the runtime are absent. Local mode remains useful offline after permitted model assets have been prepared.

## SCP-005 — Qualify named execution profiles
Required runtime profiles are `win-cpu`, `wsl-cpu`, and `wsl-cuda`, defined in [00-DECISIONS.md](00-DECISIONS.md). All mandatory lessons have a CPU path. CUDA must be selected explicitly and must fail preflight if unavailable; automatic device or precision fallback is forbidden.
The acceptance reference machines MUST have x86_64 architecture, at least 16 GiB installed RAM, at least four logical CPU cores, and at least 10 GiB free disk at the selected storage root before running the full suite. These are qualification-machine conditions, not a performance guarantee or proof that every workload fits. CUDA qualification additionally records exact GPU model, VRAM, driver, WSL/kernel version, Torch CUDA build, and available memory.
Record exact OS build, Python patch, dependency lock digest, browser version, device, and CPU thread count with qualification results. Cross-device or cross-version bit-for-bit repeatability is not promised. Same-profile deterministic controls must use fixed seeds, thread configuration, and defined numeric tolerances.
The static reader is tested in Windows Edge and Firefox stable channels; the native test receipt records the exact installed versions. WSL profiles use those Windows browsers with localhost forwarding. Browser engines, OSes, or accelerators outside this set are not qualified by this release.

## SCP-006 — Select supported model and operations
The applied backend MUST use only `HuggingFaceTB/SmolLM2-135M-Instruct` at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. This is an already instruction-tuned baseline; learner adapter training is adaptation of that baseline, not training a foundation model from random weights. Tiny-v2 owns from-scratch exercises. Their model identities, tokenizers, checkpoint schemas, and results cannot be interchanged.
The applied model is selected for bounded teaching workflows, not comparative superiority or broad answer quality. Public model metadata was checked, but local execution and dependency compatibility were not tested during specification authoring. [Tree metadata](evidence/pretrained-public-metadata.json), [revision-bound architecture/template/license evidence](evidence/model-config-and-license.json) and the [model profile](contracts/model-profile.json) freeze the selection.
The runtime MUST expose the named operations in decision D06 with distinct command and UI labels. Selecting a tokenizer cannot implicitly replace model weights. Generating, chatting, evaluating, importing, or exporting cannot implicitly start training.
A failed download, unavailable model, or unsupported artifact MUST produce an explicit error. Substituting an API model, synthetic output, different checkpoint, or random weights is forbidden.

## SCP-007 — Keep the implementation bounded
The following are excluded from this release: hosted-to-local bridges; public network listeners; remote or cloud GPU jobs; multi-user sharing/accounts; payments; deployment to a public inference endpoint; agent tools that execute model-produced commands; arbitrary Python notebooks/scripts submitted from the UI; third-party plugin execution; arbitrary model-hub IDs; quantization/QLoRA; distributed training; full-parameter fine-tuning of the pretrained baseline; RL/preference optimization; embedding-model retrieval; automatic web crawling; automatic training on chat history; an Unsloth dependency or checkpoint adapter; native macOS support.
E01 implements one defined weight-tying modification. It does not expand to a general architecture editor. E02 implements the specified local lexical retrieval experiment. It does not introduce an autonomous agent, remote document fetcher, or production RAG service.
Excluded capabilities MUST NOT have active or disabled teaser controls presented as upcoming features. The course may explain them conceptually with links to primary references.

## SCP-008 — Add explicit implementation paths
The implementation handoff MUST create or modify only the following target areas for the named purposes; this specification task does not create them:
| Target path | Responsibility |
|---|---|
| `curriculum-applied/2026-09-28/` | New Markdown lessons, manifest, answer/rubric files and source teaching fixtures |
| `dist/applied-content.json` | Generated applied content; includes edition and source digest |
| `dist/modules/` and existing frontend entrypoints/styles | Browser reader extension, forms, runtime client, rendered states, accessible interactions |
| `companion/src/llm_foundations_companion/` | Local service, worker, tiny-v2 adapter, applied model adapter, registry, evidence validation |
| `companion/pyproject.toml`, `companion/locks/` | Exact direct dependencies and per-profile transitive hash locks |
| `companion/tests/` | Contract, boundary, worker, model and persistence checks |
| `qa/browser/` | Browser journeys and accessibility fixtures/checks |
| `scripts/` | Deterministic content compilation and package checks |
| `docs/` | Setup/user/maintainer documentation and qualification records |
Source lesson files and generated applied content MUST have a verified one-to-one correspondence. The builder must fail on missing lessons, unknown IDs, duplicate IDs, broken internal links, or mismatched answers/rubrics. The existing foundation `sync_course.py` cannot silently adopt the new curriculum or replace original hash receipts.

## SCP-009 — Define data custody and network behavior
By default, learner data, checkpoints, chat transcripts, assessment, and job logs stay on the learner's computer. The local application MUST NOT include telemetry, remote error-reporting, external fonts, analytics, remote inference, or background update/download traffic.
The only network model activity is an explicitly selected `model_prepare` job for the pinned allowlist. Preparation must present identity, expected download bytes, source, and license before submission. Automatic package installation from inside the web UI is forbidden. External documentation links open only on learner activation.
Model instructions and generated text are inert text. They cannot trigger filesystem changes, shell commands, API job submissions, navigation, or executable HTML. Learner text is escaped at rendering boundaries and rejected where a contract requires numeric or enumerated values.
Imported artifacts carry imported provenance even after valid checksum/schema checks. Deterministic verification of artifact consistency must not be described as proof that a historical run happened on this computer.

## SCP-010 — Separate implementation, evidence, and qualification
Each requirement identifier is a binding obligation. Each acceptance case is a concrete required observation for its listed profiles. A feature is complete only when required cases pass against the same frozen implementation candidate; a blocked or unexecuted case remains in the denominator.
This package is specification documentation. It includes proposed APIs, schemas, fixtures, and test definitions; none are product execution evidence. Document checks may pass while all product acceptance statuses remain NOT_RUN.
The existing application's earlier 36 functional checks and source audits are historical baseline evidence. They do not qualify the new backend, applied curriculum, security boundaries, chat, model downloads, or browser interactions. The administrator browser block remains an explicit environmental limitation, not a reason to waive future native acceptance.
The delivery rules and release evidence fields are defined in [08-DELIVERY-AND-ACCEPTANCE.md](08-DELIVERY-AND-ACCEPTANCE.md).

## Decision ownership
[00-DECISIONS.md](00-DECISIONS.md) fixes shared selections. [07-INTEGRATION-CONTRACT.md](07-INTEGRATION-CONTRACT.md) resolves cross-section joins. If prose, OpenAPI, schemas, or fixtures disagree, the candidate fails specification review; implementers must not choose a convenient interpretation. Changes to model identity, operation semantics, protected source, data custody, completion criteria, dependency profiles, or acceptance denominator require a new explicit spec revision and updated trace records.

## Sources
- Local baseline: immutable commit and SHA-256 inventory above; source inspection performed 2026-09-28 local date.
- Model metadata/card: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct ; retrieved 2026-09-28 local date.
- Unsloth workflow reference only: https://unsloth.ai/docs/new/studio ; consulted 2026-09-28. Its product features do not establish compatibility with this custom tiny model.
- Accessibility standard: https://www.w3.org/TR/2024/REC-WCAG22-20241212/ ; used to define testable web acceptance, not to claim present conformance.
