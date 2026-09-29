# Contract edge review attempt 001 — harness failure

Status: **HARNESS_FAILED; no contract verdict**.

The first offline invocation stopped before any contract cases completed. The in-memory bundler rewrote `common.schema.json#/$defs/id` as `#/$defs/external__common#/$defs/id`; Draft 2020-12 resolution reported `PointerToNowhere`.

The source contracts were read only. No product, browser, network, model, training, or inference execution occurred. The validator was corrected to rewrite the fragment as `#/$defs/external__common/$defs/id`. This retained attempt must not be counted as a contract failure or PASS.
