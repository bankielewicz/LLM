# Proposed APP-009 tiny preview correction

Status: PROPOSED; owner authorization pending. This document does not amend the sealed revision 1.2 files.

Revision 1.2 is bound by `baseline.json` and `../s0/authority.json`. APP-009 at `docs/specs/intermediate-v1/06-APPLIED-MODELS.md:103` says the tiny context-preview request uses the prospective `model_id`. RUN-012 (`04-RUNTIME-AND-API.md:173`), INT-004 (`07-INTEGRATION-CONTRACT.md:94`) and the closed OpenAPI request require `checkpoint_id`. SCP decision ownership (`01-SCOPE.md:76`) and the INT header prohibit implementers from resolving this conflict by choosing one contract.

The proposed scoped amendment replaces only this APP-009 phrase:

> Its request uses `backend: "tiny"` with the prospective `model_id`, prompt, and decoding fields

with:

> Its request uses `backend: "tiny"` with the prospective `checkpoint_id`, prompt, and decoding fields

The separate amendment would make that replacement normative while preserving all existing approved files. The public request schema, checkpoint subject semantics, subject-identity field object, acceptance cases and 605-unit denominator need no change. In particular, `model_id` inside the resolved subject-identity object remains unchanged; it is distinct from the client request field.

Until authorized, tiny context preview and generation remain disabled. Tokenizer training, tiny weight training, safe resume and tiny evaluation can proceed. Draft code and development checks for the proposed correction cannot establish S2 completion or specification review PASS.
