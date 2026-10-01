# Contract edge review attempt 002 — harness failure

Status: **HARNESS_FAILED; no contract verdict**.

The second offline invocation resolved cross-file fragments, then stopped before contract cases completed because internal refs from an embedded standalone schema, such as `#/$defs/tokenizerConfig`, still pointed at the aggregate bundle root. Draft 2020-12 resolution reported `PointerToNowhere`.

The source contracts were read only. No product, browser, network, model, training, or inference execution occurred. The bundler was corrected to prefix each embedded schema's internal refs with that schema's aggregate definition path. This attempt must not be counted as a contract failure or PASS.
