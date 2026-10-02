# S2 existing-model integration plan

## Authority and starting point

S2 starts at merged S1 commit `bcb358e3c466fdf6ffce1fdefb414b6206b93f8a`, with revision 1.2 of the sealed specification. `baseline.json` binds the source, specification, acceptance registry and dependency locks. The approved authority and all 107 protected foundation files remain unchanged. Implementation lives in the isolated `codex/intermediate-s2` worktree.

This plan implements DEL-001's S2 deliverable. It is an implementation record, not a product-qualification result.

The owner approved the exact [APP-009 proposal](APP-009-amendment-proposal.md) on 2026-10-01 (America/New_York). The separate [revision 1.3 amendment](../../specs/intermediate-v1-amendments/APP-009-revision-1.3.md) defines the effective authority as the unchanged sealed revision 1.2 base plus the single approved wording replacement. [authority.json](authority.json) binds that composite authority. The original proposal remains unchanged as the reviewed record; its pending status is historical.

## Contributor gate added during this slice

Remote `main` advanced to `c5e8028dbc85d01750d633f953458e38e28fecf4` while S2 was underway. Its only change from the captured S1 baseline is `AGENTS.md`, introduced by PR #5. The source and sealed authority baseline remain unchanged. That captured guide introduced red-green-refactor work and a 95% complete-component line-coverage check. The owner subsequently supplied replacement repository instructions; S2 retains its already implemented 95% gate and reproducible evidence procedure. A source or native subset PASS cannot replace that complete-tree measurement. Instrumented coverage evidence remains separate from the locked installed-runtime gate.

## Runnable outcome

The target S2 outcome lets a learner use the existing authenticated API or owner CLI to register data, train a tokenizer, train tiny-model weights, resume a safe checkpoint, evaluate complete validation/test splits, preview prompt context and generate a continuation. The amended candidate enables tokenizer training, tiny-model training, safe resume, tiny evaluation, tiny context preview, and generation. These are distinct operations with independently checked effects. The service process imports no model framework and all compute stays in the fixed owned worker. S3 supplies the learner forms and navigation.

The target operation set is `tokenizer_train`, `tiny_train`, `tiny_resume`, tiny-subject `evaluate`, tiny-backend `context_preview`, and `generate`. All six are enabled only with a verified installed build identity and a capable runtime; backend-specific admission still rejects unavailable applied-model requests. Applied-model, chat, retrieval and bundle handlers remain unavailable. The E01 learner-source harness and its receipt-gated tied model workflow belong to the later completion slice; S2 must not accept an unverified tied-model request.

## Implementation sequence

1. **Freeze joins and validation cases.** Map INT-003 through INT-008, RUN-002/007/009/012/013, DAT-004/005 and the S2 delivery gate to concrete modules, records, tests and native observations. Resolve parent/worker input and artifact-commit interfaces before parallel implementation. Any normative contradiction stops the affected work rather than changing the sealed specification silently.
2. **Preserve model and tokenizer mechanics.** Add a companion-owned versioned copy/refactor of the protected byte/BPE tokenizer, shifted-target batching and standard TinyLM. Keep state-dictionary names, exact tokenizer fingerprint/serialization, parameter count, architecture, initialization and optimizer mechanics. Compare the shared mechanics with the unchanged lab under fixed threads/seeds.
3. **Implement safe training and resume.** Add the fixed recipe, initial/cadence/final/cancellation checkpoints, typed metrics, best/last lineage and the module-14 hold. Checkpoints use only the seven specified safetensors/JSON files. Load validates identities, tensor names/shapes/dtypes and RNG/optimizer state before effective state use. Resume creates a new run while preserving its parent.
4. **Implement evaluation and inference.** Score every record/target including EOS under the byte-normalized protocol. Context preview performs tokenizer work without loading weights. Generation verifies the matching preview, uses the specified deterministic greedy/top-p decoding and never updates weights. Imported/generated text remains inert.
5. **Integrate durable worker effects.** Validate references, digests, eligibility, lineage and checked reservations before queue insertion. Materialize verified fixed worker inputs; retain all worker writes in owned staging until parent verification. Publish checkpoint events only with durable registry commits, and commit terminal result/run/model/artifact effects consistently. Preserve cancellation, process-loss and read-only recovery behavior.
6. **Validate a frozen installed candidate.** Run source/contract/custody and S0/S1 regression checks, model equivalence, adversarial artifact tests, then actual installed service operations. Demonstrate a real tiny job, 25+25 resumed versus 50 uninterrupted updates, generation/evaluation weight immutability, preview rejection, deterministic cancellation and safe checkpoint rejection. Freeze source, wheel, interpreter and lock identities; retain every attempt outside the source worktree.
7. **Review and publish the slice.** Independently review the implemented boundaries, repair confirmed defects, commit, rerun the final candidate gate, push and create an S2 PR. Do not merge, deploy or copy changes to the primary checkout.

## Validation boundaries

Unit, service, native-model, browser and full product qualification remain separate. Native model checks require the locked runtime and real fixed workers; test doubles cannot satisfy them. The first native development target is the available WSL CPU profile. Every unexecuted profile and mandatory full-product case remains `NOT_RUN`; the full 605-unit acceptance denominator is neither reduced nor promoted by an S2 subset result. Source equivalence requires exact token IDs, targets, counts and greedy IDs, with the specified 1e-6 absolute/relative tolerance for logits/loss. Same-profile resume comparison checks tensors, batch RNG and validation values; timestamps and generated registry IDs are intentionally different.

All installed checks use a wheel built from the captured candidate. Every retry gets a new retained attempt directory. Model quality, training speed, educational effectiveness, final browser acceptance and release readiness are not established by this slice's source/service gate.
