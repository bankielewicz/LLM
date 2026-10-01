# S0 implementation review

This record describes the scoped implementation review before the retained S0 gate. It is not product qualification or owner acceptance.

## Review scope and disposition

- **Custody and gate runner:** bind both the approved revision 1.1 Git objects and active revision 1.2; tie the 107-file inventory back to the approved Git manifest; require CPython 3.12; preserve blocked mandatory gates and failed process output; use the separate pinned legacy environment. Synthetic negative controls cover receipt replacement, source changes, unrelated Git ancestry, missing files and symlink substitution.
- **Contract validator:** independently reviewed closed schema references, fixture accounting and role-specific error bindings. The review found missing pinned-model-profile validation and noncanonical JSON Pointer array indices. Both were corrected and independently rechecked.
- **Edition compiler:** independently reviewed output confinement, finite JSON numbers, closed supplement-card records, internal-link validation, fixed lesson paths, visible lesson content and deterministic fixture ordering. The follow-up also checked alternate autolinks, multiline HTML links and a symlinked fixture root. Each reported boundary was corrected and independently rechecked.
- **Dependency locks:** independently reviewed package identity, complete lock grammar, selected wheel identity and compatibility, index custody and materialization boundaries. The review found that a missing required direct package and an unrecognized active requirement line could pass. The corrected validator requires the exact frozen lock identities and rejects unsupported active syntax; negative controls cover both findings and retained partial-download failure evidence.

The authoring environment is separate from the four future runtime profiles. The final gate checks every installed authoring package against its lock. A package-drift negative control verifies this check.

## Specification corrections

AMD-055 reconciles the summary with the four already-required runtime profiles. AMD-056 represents error bindings by role, preserving the distinct 413 transport and 400 semantic uses of PAYLOAD_TOO_LARGE. AMD-057 reseals revision 1.2. The approved revision 1.1 remains recoverable from its immutable commit and exact retained manifest. The revision history points to that retained manifest.

No protected foundation file, requirement ID, acceptance case or product acceptance status was changed. All 605 product execution units remain NOT_RUN.

## Final verification boundary

Run scripts/run_s0_gate.py after freezing the source candidate. Its per-attempt result.json is the authority for the observed source gate result, candidate content digest, commit, complete gate list, exact Python-test denominator and retained output hashes. The suite uses pytest to include both unittest classes and pytest functions. Component-only test totals are not substituted for the integrated denominator.

The gate runs the unchanged original JavaScript and Python lab regressions and the historical source audit. Legacy training, resume, evaluation and generation establish foundation compatibility only. They do not establish companion service behavior or intermediate model qualification.

Later S1-S5 implementation, four-profile installation and fit calibration, the product-produced device comparison snapshot, native browser accessibility and learner assessment remain separate work.
