# Evidence verification mapping review

Status: closed by the normative [evidence verification contract](../contracts/evidence-verification.md). This review records the gaps found during curriculum-to-API mapping and their binding resolution. It is not a second source of verifier behavior.

## Findings and closure

1. **Foundation lesson 12 had no trustworthy path from preserved lab files to objective companion evidence.** Closed by requiring a companion-created tiny-v2 reenactment of the preserved capstone operations. External lab files remain unverified claims; no arbitrary legacy checkpoint loader is introduced.
2. **Rejected configuration and tampered-bundle attempts lacked immutable evidence.** Closed by the two-form `POST /diagnostic-receipts` contract: prescribed `tiny_head_validation`, or a receipt anchored to an existing failed bundle validation/import job. The endpoint creates a diagnostic artifact only.
3. **The E01 CLI receipt appeared unregistrable through HTTP.** Closed by the existing current-user control channel in RUN-001/APP-014. No browser/HTTP receipt import is permitted because that would weaken custody of local source verification.
4. **Module-15 error categories and E02 citation choices were narrative-only.** Module-15 categories/spans remain assessed narrative. E02 receives the single closed `retrieval-citation-submission-v1` structured response with fixed query order and UTF-8 half-open scalar-boundary offsets.
5. **The capstone plan could not bind an immutable note revision.** Closed by `note_refs:[{note_id,revision}]`; evidence creation snapshots the exact note bytes/digest. CUR-012 accepts exactly one plan note reference.
6. **`intermediate_evidence_packet_complete` had no owner.** Closed as a server-derived read-only field that is true only for verified CUR-012 evidence.
7. **The umbrella `intermediate-required-v1` profile could not select a finite verifier contract.** Closed by item-specific profiles for CUR-005 through CUR-012 plus `prerequisite-p00-v1`; foundation and elective profile names remain stable.
8. **Dataset eligibility had been described as booleans in early prose.** Closed against the authoritative `data-audit.schema.json` values: clean is `eligible`; the deliberately leaky fixture is `audit_only`; `rejected` remains the third schema value.

No product behavior was executed in this review. Verification cases remain `NOT_RUN` until implementation and native qualification.
