# Round 01 — root contract review
Date: 2026-09-28. Reviewer: independent spec_cross_review agent, followed by root integration review. Scope: authoring drafts, not product execution.
All findings below are retained even after remediation. Final disposition belongs to round-02 and final review records.

| ID | Severity | Finding | Required resolution |
|---|---|---|---|
| R1-01 | HIGH | DEL-003 permits carrying passes to a changed implementation while SCP-010 requires one frozen candidate. | Every implementation/build/lock/model/contract/fixture/registry change creates a new candidate with all mandatory execution rows NOT_RUN. |
| R1-02 | HIGH | SCP trace labels behavior journeys as source checks. | Classify browser and integration checks accurately; split mixed-mode cases. |
| R1-03 | HIGH | Static mode missing from full native accessibility case. | Include static in browser/profile denominator. |
| R1-04 | HIGH | Limits omit notes, chat, prompts, logs/events, archives and total registry/storage. | Publish authoritative bounded limit/rejection map and align schemas/API. |
| R1-05 | MEDIUM | Human and self assessment labels can imply independent verification. | Freeze explicit self/independent kind, reviewer identity and truthful labels. |
| R1-06 | MEDIUM | Saved model tree evidence does not substantiate declared architecture/license. | Retain revision-bound config and license metadata with URLs/digests. |
| R1-07 | MEDIUM | Baseline hash case can accept a rebound source receipt. | Anchor source receipt to immutable base commit/tree and exact receipt digest. |

Root's additional review challenged: bundle registration missing after validation; independent progress flags; static hash routes and old-route alias; exact tokenizer versus structural simulation labels; preview size; duplicate import policy; model redirect allowlist; queue/event cardinality; reload/pairing; fixed diagnosis hold expiry; curriculum/data task parser; retrieval IDs/ranges; applied resume equality; supervised EOS mask; decoding RNG; weight-tying learner-code harness boundary; sealed-test exposure across attempts; complete authored fixture text.
These are design-document findings. No product bug was executed or fixed in this task.
