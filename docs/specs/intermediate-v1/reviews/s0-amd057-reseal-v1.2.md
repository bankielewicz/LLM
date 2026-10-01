# S0 AMD-057 — reseal specification revision 1.2

Status: SEALED on 2026-10-01 under the owner's scoped direction to proceed. This seal establishes specification custody only. It does not establish product implementation, qualification, release, or owner acceptance of an implementation candidate.

## Predecessor custody

Revision 1.1 remains in Git at commit acb04d173ab1c4c6395ac95c95ad04b8fa50e849. Its exact manifest is retained at docs/implementation/s0/approved-1.1-manifest.json with file SHA-256 13645c0e25f45f1d478a34d79fec1affdc4e06304cc304f03967b2194708adaa. That manifest binds 150 normative files with payload SHA-256 f611dddea0addf4da6329b2157a8adbb0458614d92c59b9341610921f4918e62.

## Revision 1.2 seal

The revision-1.2 manifest has file SHA-256 7526190e853580adcc3681c9ac0559be4ae8c741878b5a686bdcf624d9f131b8. It binds 150 normative files with payload SHA-256 974306b73634d9d073fb90de941fe55c8fb83cd073a585613cb6249b06691571. The normative file count is unchanged.

## Final checks

- Document/index check: PASS in reviews/s0-amd057-authority-link-document-v1.2.json — 108 requirements, 128 cases, 605 execution units, 31 schemas and 50 schema fixtures.
- Strict contract-edge check: PASS in reviews/s0-amd057-authority-link-contract-edges-v1.2.md.
- Fixture-oracle check: PASS in reviews/s0-amd057-authority-link-fixture-oracles-v1.2.json — 155 cases.
- Fixture materialization check: PASS for 14 fixtures under Python 3.12.3; manifest SHA-256 030ff8f771767f319f1abd6b3702891f243a7ef15070e8fc89dda40590a5e4f0.
- Seal check: PASS — 150 files, payload SHA-256 974306b73634d9d073fb90de941fe55c8fb83cd073a585613cb6249b06691571.
- Whitespace check: PASS.

All 605 product acceptance execution units remain NOT_RUN.
