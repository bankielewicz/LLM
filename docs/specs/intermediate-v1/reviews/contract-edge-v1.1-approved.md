# Intermediate v1 contract edge checks

This is an offline specification-contract review. It executed Python `Draft202012Validator` checks and small semantic oracles only. It did not start the companion, execute product Python, train or run a model, open a browser, use the network, or assess rendered accessibility.

## Outcome

- Executable checks: **82 passed, 0 failed, 82 total**.
- Open findings: **0**.
- Browser, native runtime, installation, training, inference, and accessibility acceptance: **NOT_RUN**.

Schema compilation alone is not treated as useful validation. The checks also exercise every job request/result discriminator, normal runtime/job states, preview and capstone token binding, assessment edge cases, static module IDs, declared fixtures, and cross-file references.

## Source identity

| Artifact | SHA-256 / value |
|---|---|
| openapi.json | `e0602505eb4f9381695950c2ed1610ec74e0a0166014de6006e1dc05e2053585` |
| model-profile.json | `9cbb21f94816007a72c2fcf9073edbf34a4fcefcc45f52879353cd9faeb0be0a` |
| capstone-rubric.json | `04f6ea803ccd5116fe14cc3a2875364e0dad12598d47b47d7268f4e198876f91` |
| schema_count | `30` |
| ref_count | `683` |

## Open findings

No open contract findings were reproduced.

## Executed cases

| Case | Status | Observation |
|---|---|---|
| `CEC-SCHEMA-001` | **PASS** | 32 Draft 2020-12 schema units compile |
| `CEC-SCHEMA-002` | **PASS** | 683 local/external refs resolve |
| `CEC-API-001` | **PASS** | request, result and Job operation sets are the same closed 15-operation set |
| `CEC-REQ-ADAPTER_RESUME` | **PASS** | normal adapter_resume request matches JobRequest and exactly one branch (branches=1) |
| `CEC-RES-ADAPTER_RESUME` | **PASS** | normal adapter_resume result matches JobResult and exactly one branch (branches=1) |
| `CEC-REQ-ADAPTER_TRAIN` | **PASS** | normal adapter_train request matches JobRequest and exactly one branch (branches=1) |
| `CEC-RES-ADAPTER_TRAIN` | **PASS** | normal adapter_train result matches JobResult and exactly one branch (branches=1) |
| `CEC-REQ-CHAT_GENERATE` | **PASS** | normal chat_generate request matches JobRequest and exactly one branch (branches=1) |
| `CEC-RES-CHAT_GENERATE` | **PASS** | normal chat_generate result matches JobResult and exactly one branch (branches=1) |
| `CEC-REQ-CONTEXT_PREVIEW` | **PASS** | normal context_preview request matches JobRequest and exactly one branch (branches=1) |
| `CEC-RES-CONTEXT_PREVIEW` | **PASS** | normal context_preview result matches JobResult and exactly one branch (branches=1) |
| `CEC-REQ-EVALUATE` | **PASS** | normal evaluate request matches JobRequest and exactly one branch (branches=1) |
| `CEC-RES-EVALUATE` | **PASS** | normal evaluate result matches JobResult and exactly one branch (branches=1) |
| `CEC-REQ-EXPORT_BUNDLE` | **PASS** | normal export_bundle request matches JobRequest and exactly one branch (branches=1) |
| `CEC-RES-EXPORT_BUNDLE` | **PASS** | normal export_bundle result matches JobResult and exactly one branch (branches=1) |
| `CEC-REQ-GENERATE` | **PASS** | normal generate request matches JobRequest and exactly one branch (branches=1) |
| `CEC-RES-GENERATE` | **PASS** | normal generate result matches JobResult and exactly one branch (branches=1) |
| `CEC-REQ-IMPORT_BUNDLE` | **PASS** | normal import_bundle request matches JobRequest and exactly one branch (branches=1) |
| `CEC-RES-IMPORT_BUNDLE` | **PASS** | normal import_bundle result matches JobResult and exactly one branch (branches=1) |
| `CEC-REQ-MODEL_PREPARE` | **PASS** | normal model_prepare request matches JobRequest and exactly one branch (branches=1) |
| `CEC-RES-MODEL_PREPARE` | **PASS** | normal model_prepare result matches JobResult and exactly one branch (branches=1) |
| `CEC-REQ-RETRIEVAL_BUILD` | **PASS** | normal retrieval_build request matches JobRequest and exactly one branch (branches=1) |
| `CEC-RES-RETRIEVAL_BUILD` | **PASS** | normal retrieval_build result matches JobResult and exactly one branch (branches=1) |
| `CEC-REQ-RETRIEVAL_QUERY` | **PASS** | normal retrieval_query request matches JobRequest and exactly one branch (branches=1) |
| `CEC-RES-RETRIEVAL_QUERY` | **PASS** | normal retrieval_query result matches JobResult and exactly one branch (branches=1) |
| `CEC-REQ-TINY_RESUME` | **PASS** | normal tiny_resume request matches JobRequest and exactly one branch (branches=1) |
| `CEC-RES-TINY_RESUME` | **PASS** | normal tiny_resume result matches JobResult and exactly one branch (branches=1) |
| `CEC-REQ-TINY_TRAIN` | **PASS** | normal tiny_train request matches JobRequest and exactly one branch (branches=1) |
| `CEC-RES-TINY_TRAIN` | **PASS** | normal tiny_train result matches JobResult and exactly one branch (branches=1) |
| `CEC-REQ-TOKENIZER_TRAIN` | **PASS** | normal tokenizer_train request matches JobRequest and exactly one branch (branches=1) |
| `CEC-RES-TOKENIZER_TRAIN` | **PASS** | normal tokenizer_train result matches JobResult and exactly one branch (branches=1) |
| `CEC-REQ-VALIDATE_BUNDLE` | **PASS** | normal validate_bundle request matches JobRequest and exactly one branch (branches=1) |
| `CEC-RES-VALIDATE_BUNDLE` | **PASS** | normal validate_bundle result matches JobResult and exactly one branch (branches=1) |
| `CEC-API-002` | **PASS** | normal GET /api/v1/runtime response validates with every fixed limit |
| `CEC-GPU-OFFER` | **PASS** | GPU offer statuses carry exactly the GPU and reason fields RUN-001 allows |
| `CEC-STATE-QUEUED` | **PASS** | normal queued Job representation validates |
| `CEC-STATE-STARTING` | **PASS** | normal starting Job representation validates |
| `CEC-STATE-RUNNING` | **PASS** | normal running Job representation validates |
| `CEC-STATE-CANCELLING` | **PASS** | normal cancelling Job representation validates |
| `CEC-STATE-COMPLETED` | **PASS** | normal completed Job representation validates |
| `CEC-STATE-FAILED` | **PASS** | normal failed Job representation validates |
| `CEC-STATE-INTERRUPTED` | **PASS** | normal interrupted Job representation validates |
| `CEC-EVENT-001` | **PASS** | normal state_changed event validates |
| `CEC-PREVIEW-001` | **PASS** | tiny and chat preview examples validate and bind to their generation requests |
| `CEC-PREVIEW-002` | **PASS** | stale changed request remains shape-valid but the executable digest check rejects it |
| `CEC-TOKEN-001` | **PASS** | capstone create/created shapes validate and the one-time token matches its stored SHA-256 and evaluate request |
| `CEC-RUBRIC-001` | **PASS** | 10-row rubric and normal self/independent assessment records validate |
| `CEC-RUBRIC-002` | **PASS** | assessment request without reviewer is rejected |
| `CEC-RUBRIC-003` | **PASS** | assessment request with a score outside 0/5/10 is rejected |
| `CEC-STATIC-001` | **PASS** | all 24 static module IDs and independent progress flags validate |
| `CEC-STATIC-002` | **PASS** | out-of-range static module ID 21 is rejected |
| `CEC-STATIC-003` | **PASS** | static installation UUID, independent progress, and portable learning-state wrapper validate |
| `CEC-STATIC-004` | **PASS** | all 20 module references in curriculum-overlay.json are recognized |
| `CEC-FIXTURE-001` | **PASS** | all 49 declared data validation fixtures match expectation |
| `CEC-FIXTURE-002` | **PASS** | applied pinned model profile validates against model-profile.json |
| `CEC-SEMANTIC-001` | **PASS** | relative eval cadence is shape-valid but rejected by the declared executable semantic invariant |
| `CEC-EVIDENCE-CUR-004` | **PASS** | module submission and verification are representable; actual artifact checks remain native |
| `CEC-EVIDENCE-FOUNDATION-12` | **PASS** | module submission and verification are representable; actual artifact checks remain native |
| `CEC-EVIDENCE-CUR-005` | **PASS** | module submission and verification are representable; actual artifact checks remain native |
| `CEC-EVIDENCE-CUR-006` | **PASS** | module submission and verification are representable; actual artifact checks remain native |
| `CEC-EVIDENCE-CUR-007` | **PASS** | module submission and verification are representable; actual artifact checks remain native |
| `CEC-EVIDENCE-CUR-008` | **PASS** | module submission and verification are representable; actual artifact checks remain native |
| `CEC-EVIDENCE-CUR-009` | **PASS** | module submission and verification are representable; actual artifact checks remain native |
| `CEC-EVIDENCE-CUR-010` | **PASS** | module submission and verification are representable; actual artifact checks remain native |
| `CEC-EVIDENCE-CUR-011` | **PASS** | module submission and verification are representable; actual artifact checks remain native |
| `CEC-EVIDENCE-CUR-012` | **PASS** | module submission and verification are representable; actual artifact checks remain native |
| `CEC-EVIDENCE-CUR-014` | **PASS** | module submission and verification are representable; actual artifact checks remain native |
| `CEC-EVIDENCE-CUR-015` | **PASS** | module submission and verification are representable; actual artifact checks remain native |
| `CEC-EVIDENCE-RETIRED` | **PASS** | ambiguous umbrella verification profile is rejected |
| `CEC-EVIDENCE-CITATION` | **PASS** | six citations require exact query IDs in fixed order |
| `CEC-DIAGNOSTIC` | **PASS** | negative teaching request can reach pure validation receipt path without a compute job |
| `CEC-CONVERSATION-BASE` | **PASS** | create and storage share the same subject contract |
| `CEC-CONVERSATION-ADAPTER` | **PASS** | create and storage share the same subject contract |
| `CEC-BUNDLE-OPTIONS` | **PASS** | all content inclusion choices must be explicit |
| `CEC-BUNDLE-READINESS` | **PASS** | bundle readiness uses ready/base_required/metadata_only only |
| `CEC-ASSESSMENT-80-1` | **PASS** | explicit score/gate boundary matches semantic-rules derivation |
| `CEC-ASSESSMENT-85-1` | **PASS** | explicit score/gate boundary matches semantic-rules derivation |
| `CEC-ASSESSMENT-100-0` | **PASS** | explicit score/gate boundary matches semantic-rules derivation |
| `CEC-ASSESSMENT-50-1` | **PASS** | explicit score/gate boundary matches semantic-rules derivation |
| `CEC-ASSESSMENT-45-1` | **PASS** | explicit score/gate boundary matches semantic-rules derivation |
| `CEC-PREFLIGHT-CHILD` | **PASS** | fixed child report is representable without model execution |
| `CEC-PREFLIGHT-CHILD-CUDA` | **PASS** | CUDA child reports record device facts; a passed result cannot carry a reason code or omit the device |

## Boundary interpretation

- `CEC-PREVIEW-002` proves the JSON shape alone cannot detect a stale preview. The separately executed digest oracle rejects the changed request, matching RUN-011; the implementation still needs a native test.
- `CEC-TOKEN-001` checks the raw one-time capstone token against the stored SHA-256 and the sealed evaluate request. It does not prove atomic consumption or replay rejection.
- The state examples establish representability. They do not prove transactions, ordering, cancellation, recovery, or worker behavior.
- Static backup checks establish schema compatibility only. They do not exercise localStorage quotas, browser file pickers, migration, or reload.

## Reproduction

From the specification repository root:

```bash
python3 docs/specs/intermediate-v1/tools/check_contract_edges.py --report reviews/<fresh-name>.md
```

The tool refuses to overwrite an existing report. It returns nonzero when any executable check fails or any finding remains open. Use `--allow-findings` only when intentionally recording a review attempt with known open findings; the report still lists them.
