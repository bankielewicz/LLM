# APP-009 tiny preview correction — effective revision 1.3

Status: NORMATIVE SCOPED AMENDMENT. The owner approved the exact proposal at `docs/implementation/s2/APP-009-amendment-proposal.md`, SHA-256 `88e4458774e3c0fa38e92042d1c604fc2804b09dc8b396bcdf509ab425189a7c`, on 2026-10-01.

## Authority composition

Effective specification revision 1.3 is the sealed revision 1.2 package plus this amendment. The revision 1.2 package remains sealed at 150 files with manifest SHA-256 `7526190e853580adcc3681c9ac0559be4ae8c741878b5a686bdcf624d9f131b8` and payload SHA-256 `974306b73634d9d073fb90de941fe55c8fb83cd073a585613cb6249b06691571`. This amendment does not replace, rewrite, or reseal that package.

The S0 authority, S2 baseline, and approved proposal remain unchanged. The S2 amendment-authority sidecar binds their exact bytes, this document, and the substitution below.

## Normative substitution

In the sealed revision 1.2 file `docs/specs/intermediate-v1/06-APPLIED-MODELS.md`, APP-009 contains this phrase exactly once:

> Its request uses `backend: "tiny"` with the prospective `model_id`, prompt, and decoding fields

For the effective revision 1.3 authority, read that phrase as:

> Its request uses `backend: "tiny"` with the prospective `checkpoint_id`, prompt, and decoding fields

This is the only normative substitution made by this amendment.

## Preserved contracts and status

The public `ContextPreviewRequest` already requires `checkpoint_id` for the tiny backend. RUN-012 and INT-004 already require a tiny checkpoint subject. The `model_id` inside the resolved subject-identity object remains unchanged because it is a server-resolved identity field, distinct from the client request field corrected here.

No OpenAPI, schema, fixture, requirement, trace, acceptance-case, dependency, protected-source, or sealed revision 1.2 byte changes. The 605 product acceptance execution units remain in their complete denominator with status `NOT_RUN`. This amendment supplies normative authority for implementation and qualification of the corrected request field; it does not establish product, profile, browser, or learner acceptance.
