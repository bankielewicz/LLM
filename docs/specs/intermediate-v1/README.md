# LLM Foundations — intermediate buildout specification

Revision: 1.1 — approved by the owner on 2026-09-30. Assembled 2026-09-30 by an independent specification review from revision 1.0 (commit 0f5bc7c3e6d0dc572530b612e667367d7a164750, payload SHA-256 ccc00c3e3ab78439866617ea750fd176192e8b40c87a3f5914706d39584f702a) and amendments AMD-001 through AMD-053 plus the reseal, AMD-054, all approved by the owner. Revision 1.1 is the sealed authority; revision 1.0's seal and its reviews remain historical. Status: implementation specification; product acceptance is NOT_RUN.

This package specifies the expansion of **LLM Foundations: build, measure, explain** into an application that teaches an adult beginner to conduct a bounded, reproducible applied LLM experiment. It preserves the supplied thirteen-lesson foundation and Python labs. It adds eight required applied modules, a prerequisite refresher, and two shipped electives, for 24 addressable modules.

The build has two modes: a static learning application and a same-origin local Python companion. The static Site does not run training or inference. The companion provides actual tokenizer training, tiny-model training/resume/evaluation, a pinned pretrained baseline, adapter training/resume, context preview, continuation/chat, retrieval, and portable artifacts. These operations have separate identities and effects.

## Start here
Read the documents in numeric order. The implementation order and release gate are in 08; the concrete joins and limits are in 07. A prose/schema/fixture conflict blocks implementation qualification and requires an explicit amendment.

| Document | Binding content |
|---|---|
| [00 — Decisions](00-DECISIONS.md) | Audience, edition, modes, profiles, pinned choices, scope boundaries |
| [01 — Scope](01-SCOPE.md) | Immutable source baseline,24 modules, supported/excluded behavior, target source paths |
| [02 — Curriculum](02-CURRICULUM.md) | Every new module's outcome, examples, exercises, evidence, feedback, stopping point and rubric |
| [03 — Learner experience](03-LEARNER-EXPERIENCE.md) | Routes, screens, state transitions, visual tokens, search, responsive/keyboard/accessibility behavior |
| [04 — Runtime and API](04-RUNTIME-AND-API.md) | Local launch, authentication, worker ownership, queue, jobs/events, errors, storage and recovery |
| [05 — Data and artifacts](05-DATA-AND-ARTIFACTS.md) | Closed schemas, audit/splits, identity, lineage, progress, notes, import/export, atomicity |
| [06 — Applied models](06-APPLIED-MODELS.md) | Exact model revision/assets, masks, LoRA recipes, deterministic decoding, chat, retrieval, weight tying |
| [07 — Integration](07-INTEGRATION-CONTRACT.md) | UI/API joins, TinyLM mechanics/checkpoints, full byte-NLL protocol, limits, provenance and amendments |
| [08 — Delivery and acceptance](08-DELIVERY-AND-ACCEPTANCE.md) | Dependency-ordered slices, installable artifacts, frozen qualification denominator, real learner journeys |

Machine contracts:
- [OpenAPI 3.1](contracts/openapi.json) defines local HTTP requests, responses and discriminated jobs.
- [Evidence verification mapping](contracts/evidence-verification.md) fixes each module’s objective evidence, structured submission, and assessment boundary.
- [Semantic rules](contracts/semantic-rules.json) fixes arithmetic and server-derived assertions that JSON Schema alone cannot enforce.
- [Data/schema directory](contracts/schemas) defines durable records and portable formats.
- [Model-profile schema](contracts/model-profile.json) and [pinned profile instance](fixtures/applied/smollm2-135m-instruct-v1.json) fix the applied baseline.
- [Requirement registry](requirements.json) and [acceptance registry](acceptance-cases.json) are generated from section traces. Every product case and expanded profile/browser row begins NOT_RUN.
- [Curriculum fixtures](fixtures/curriculum), [authored data](fixtures/data), and [applied oracles](fixtures/applied) fix the teaching inputs and expected arithmetic/protocol behavior.

## What completion means
Learners must supply foundation-capstone evidence and the objective evidence for modules 13–20, then apply the explicit reasoning rubric. Read, practiced, self-checked, artifact verified, and human review remain separate. Self-review and independent review have different labels. The app cannot infer understanding from navigation, a completed training process, or a low loss value.

The required capstone compares the frozen pretrained base with a specified adapter on the new support task, retains negative results, records test exposure, and demonstrates bundle reuse. E01 assesses a scoped learner source modification using an explicit local CLI harness; the browser never executes uploaded Python. E02 teaches lexical retrieval with source evidence. Neither elective substitutes for required core evidence.

The reference pretrained model is HuggingFaceTB/SmolLM2-135M-Instruct at revision 12fd25f77366fa6b3b4b768ec3050bf629380bac. This is an already instruction-tuned model. LoRA adaptation is separate from the TinyLM from-scratch path. The release requires Windows CPU, WSL CPU, and explicitly selected WSL CUDA qualification; CPU can complete every required lesson. Dependency pins and model identity are selected requirements, not tested compatibility or performance claims.

## Authoring custody and review
The work is isolated on local branch spec/intermediate-buildout in the sibling llm-foundations-buildout-spec worktree. The base is commit 3a47ea48da53cb9de3ff4727ca5f0f0b6f2b9bf8. Application/course bytes are outside this task's authoring scope.

Five specialist authors owned curriculum, UX, runtime, data, and applied-model sections. A sixth agent independently reviewed the drafts; root reviewed and integrated their work. Retained review rounds include:
- [Round 01 root-contract findings](reviews/round-01-root.md).
- [Round 02 independent cross-section findings](reviews/round-02-independent.md).
- [Contract example checks](reviews/contract-edge-checks.md).

The [final review](reviews/FINAL-REVIEW.md), [root closure](reviews/round-03-root-closure.md), and [payload manifest](spec-manifest.json) accompany the completed package. Earlier failed checks are retained. A document check does not qualify application behavior.

The authoring process used public model/config/license and dependency metadata, authored fixtures, schema validators, source hashing, and contract inspection. It did not install the selected runtime, download model weights, train/adapt models, run native browser journeys, publish a Site, or modify the original labs. Browser execution remains unavailable under administrator policy; its future acceptance cases are mandatory and cannot be waived by a source audit.

## Revision history
| Revision | State | Payload | Content |
|---|---|---|---|
| 1.0 | Sealed at commit 0f5bc7c3e6d0dc572530b612e667367d7a164750 | 117 files, SHA-256 ccc00c3e3ab78439866617ea750fd176192e8b40c87a3f5914706d39584f702a | Original package; the reports under reviews/ describe revision 1.0 only. |
| 1.1 | Approved by the owner 2026-09-30; sealed | [spec-manifest.json](spec-manifest.json) | Revision 1.0 plus AMD-001 through AMD-053 and this reseal, AMD-054, from the independent review of 2026-09-30, applied in amendment-index order; each amendment's unified patch is retained with that review. |

## Reproduce document checks
Use a Python environment with jsonschema and referencing already available. Revision 1.0 was authored with a Windows Python 3.10 interpreter; from revision 1.1 every document check, including fixture materialization, MUST pass under the qualified Python 3.12 interpreter, which is also the companion target. A check that passes only under another interpreter does not satisfy the S0 gate.

From the worktree root:

```text
python docs/specs/intermediate-v1/tools/check_spec.py --report reviews/document-check-new-attempt.json --check-index
python docs/specs/intermediate-v1/tools/check_contract_edges.py --strict --report reviews/contract-edge-new-attempt.md
python docs/specs/intermediate-v1/tools/check_fixture_oracles.py --report reviews/fixture-oracles-new-attempt.json
python docs/specs/intermediate-v1/fixtures/data/generate_fixtures.py --check
python docs/specs/intermediate-v1/tools/seal_spec.py --check
```

Each retained report path must be new. check_spec validates strict JSON, schema/reference structure, schema and semantic fixture cases, local links, requirement/case/profile coverage, byte equality of requirements.json and acceptance-cases.json with the index generated from prose and trace files (`--check-index`, which never writes), OpenAPI structure and all 107 pre-existing file hashes. `--write-index --revision X.Y` is an authoring command that rewrites the two registries; it is not a check. The seal excludes exactly `reviews/**` at the specification root, `spec-manifest.json`, `__pycache__` directories and `.pyc` files. The edge checker probes example contracts; it does not execute HTTP requests or model code. Native/model/browser qualification follows 08 after implementation.

The [source inventory](evidence/source-baseline.json), [model tree metadata](evidence/pretrained-public-metadata.json), [revision-bound config/template/license](evidence/model-config-and-license.json), [direct dependency metadata](evidence/dependency-public-metadata.json), and the [wsl-cpu verification record](evidence/qualification-wsl-cpu-2026-09-30.json) (dependency set, all eight model-file digests and redirect chain, adapter format, indicative timings) are evidence with bounded meanings. Metadata existence, source preservation, and schema success are not runtime compatibility or educational-effectiveness evidence.
