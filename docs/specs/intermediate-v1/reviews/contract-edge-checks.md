# Contract example review history

These are offline schema and arithmetic checks, not application execution.

- Attempts 001–003 retained checker construction failures.
- [Attempt 004](contract-edge-attempt004.md): 53 checks passed, one failed; nine specification findings retained.
- [Attempt 005](contract-edge-attempt005.md): 55 checks passed, zero failed; selected findings closed by root.
- [Attempt 006](contract-edge-attempt006.md): 55 checks passed, zero failed after bundle/checkpoint/conversation consistency corrections; 23 schema fixtures included.

The final root closure and package validation reports record any later checks. Zero findings from this checker means none of its selected counterexamples reproduced; it is not a proof that every implementation question has been eliminated.

- [Attempt007](contract-edge-attempt007.md): 65/65 passed after additional portability/assessment/preflight examples.
- [Attempt008](contract-edge-attempt008.md): curriculum author independently invoked the same 65-case checker; 65/65 passed.
- [Attempt009](contract-edge-attempt009.md): 80/80 passed including all twelve evidence submissions and rejected retired/duplicate-citation inputs.

- [Attempt010](contract-edge-attempt010.md): final candidate, 80/80 passed, zero reproduced open findings.
