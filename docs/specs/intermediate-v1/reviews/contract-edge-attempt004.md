# Intermediate v1 contract edge checks

This is an offline specification-contract review. It executed Python `Draft202012Validator` checks and small semantic oracles only. It did not start the companion, execute product Python, train or run a model, open a browser, use the network, or assess rendered accessibility.

## Outcome

- Executable checks: **53 passed, 1 failed, 54 total**.
- Open findings: **9**.
- Browser, native runtime, installation, training, inference, and accessibility acceptance: **NOT_RUN**.

Schema compilation alone is not treated as useful validation. The checks also exercise every job request/result discriminator, normal runtime/job states, preview and capstone token binding, assessment edge cases, static module IDs, declared fixtures, and cross-file references.

## Source identity

| Artifact | SHA-256 / value |
|---|---|
| openapi.json | `43e91036c6bf7bed1edbb3ba96730bdfb3238e34b73447bba945b365a38bf071` |
| model-profile.json | `ebdb413bc5d77264c1903eda713eced63c7f7795d9373354377d7bf5e1938a40` |
| capstone-rubric.json | `04f6ea803ccd5116fe14cc3a2875364e0dad12598d47b47d7268f4e198876f91` |
| schema_count | `21` |
| ref_count | `443` |

## Open findings

### CEC-F001 · P1 · The supplied valid assessment does not identify the shipped rubric rows

valid-assessment.json uses CR-01, CR-02, CR-03, CR-04, CR-05, CR-06, CR-07, CR-08, CR-09, CR-10; capstone-rubric.json uses problem_framing, data_custody, baseline, candidate_control, operation_distinction, evaluation_discipline, diagnosis, reproducibility, reuse_safety, claim_quality. The generic criterion_id schema accepts both, so the declared valid fixture can be stored without referring to any actual rubric row.

### CEC-F002 · P1 · AssessmentRequest accepts a value the stored assessment contract rejects

OpenAPI AssessmentRequest rationale has minLength 0, while assessment-record.schema.json requires minLength 1. A normal accepted empty rationale cannot be represented by the response/storage contract.

### CEC-F003 · P1 · Assessment schemas allow duplicate and omitted rubric criteria

uniqueItems compares whole row objects. Two rows with the same criterion_id and different comments validate, so one required rubric criterion may be omitted while the request still contains 10 rows.

### CEC-F004 · P1 · Assessment decision is not constrained by rubric score and objective gates

A request with ten zero scores and decision meets validates. The shipped rubric requires objective gates, total at least 80, and no zero rows.

### CEC-F005 · P1 · Job schema accepts state combinations forbidden by the runtime contract

Accepted impossible examples: queued_with_started_at, completed_without_result, failed_without_error. Runtime requires queued jobs not to have started and completion to have a typed result; failed jobs require failure data.

### CEC-F006 · P1 · Job operation is not bound to its typed result discriminator

A completed Job with operation generate and a tokenizer_train result validates. Runtime treats discriminator mismatch as WORKER_PROTOCOL_ERROR.

### CEC-F007 · P1 · Job event_type is not bound to its payload variant

A state_changed event carrying ProgressPayload validates because payload is an uncorrelated oneOf. Consumers cannot safely dispatch on event_type.

### CEC-F008 · P2 · TinyTrainRequest omits the eval_every no-greater-than-steps invariant

steps=5 and eval_every=10 validates although RUN-011 requires eval_every no greater than steps before job creation.

### CEC-F009 · P2 · ModelPrepareResult does not bind the fixed file manifest to total_bytes

Changing one authoritative file byte count to zero still validates while total_bytes remains 272437573. Runtime text requires eight exact files, sizes, digests, and exact total.

## Executed cases

| Case | Status | Observation |
|---|---|---|
| `CEC-SCHEMA-001` | **PASS** | 23 Draft 2020-12 schema units compile |
| `CEC-SCHEMA-002` | **PASS** | 443 local/external refs resolve |
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
| `CEC-TOKEN-001` | **FAIL** | capstone create/created shapes validate and the one-time token matches its stored SHA-256 and evaluate request |
| `CEC-RUBRIC-001` | **PASS** | 10-row rubric and normal self/independent assessment records validate |
| `CEC-RUBRIC-002` | **PASS** | assessment request without reviewer is rejected |
| `CEC-RUBRIC-003` | **PASS** | assessment request with a score outside 0/5/10 is rejected |
| `CEC-STATIC-001` | **PASS** | all 24 static module IDs and independent progress flags validate |
| `CEC-STATIC-002` | **PASS** | out-of-range static module ID 21 is rejected |
| `CEC-STATIC-003` | **PASS** | static installation UUID, independent progress, and portable learning-state wrapper validate |
| `CEC-STATIC-004` | **PASS** | all 20 module references in curriculum-overlay.json are recognized |
| `CEC-FIXTURE-001` | **PASS** | all 12 declared data validation fixtures match expectation |
| `CEC-FIXTURE-002` | **PASS** | applied pinned model profile validates against model-profile.json |

## Boundary interpretation

- `CEC-PREVIEW-002` proves the JSON shape alone cannot detect a stale preview. The separately executed digest oracle rejects the changed request, matching RUN-011; the implementation still needs a native test.
- `CEC-TOKEN-001` checks the raw one-time capstone token against the stored SHA-256 and the sealed evaluate request. It does not prove atomic consumption or replay rejection.
- The state examples establish representability. They do not prove transactions, ordering, cancellation, recovery, or worker behavior.
- Static backup checks establish schema compatibility only. They do not exercise localStorage quotas, browser file pickers, migration, or reload.

## Reproduction

From the specification repository root:

```bash
python3 docs/specs/intermediate-v1/tools/check_contract_edges.py --report docs/specs/intermediate-v1/reviews/<fresh-name>.md
```

The tool refuses to overwrite an existing report. It returns nonzero when any executable check fails or any finding remains open. Use `--allow-findings` only when intentionally recording a review attempt with known open findings; the report still lists them.
