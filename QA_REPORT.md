# Learner interface verification — 2026-09-26

## Result

The static application is implemented and passes the non-browser checks below. It is not fully browser-qualified: the administrator-enforced browser policy rejected inspection of the local preview. No alternate browser, indirect rendering, or screenshot workaround was used.

## Executed checks

- JavaScript syntax: `app.js` and `core.js` passed Node syntax checks.
- 24 functional tests passed. These cover all six supplied byte/BPE reference cases (exact IDs and reconstructed text), further whitespace/Unicode cases, partial Unicode-byte display, merge validation, shifted targets, future-token exclusion from earlier contexts, tensor shapes, vocabulary-matrix growth, both recorded examples and their SHA-256 tokenizer identities, malformed/incompatible imports, failed/interrupted/missing statuses, progress persistence, storage failure, backup merging, capstone-reference validation, all canonical lessons, Windows environment commands, and text escaping.
- 34 page templates evaluated successfully as HTML strings without a browser: home, roadmap, every lesson, four explorer views, recorded results, notebook, capstone, labs, checkpoint journey, glossary, and supplied reference documents. Additional template checks cover no-data results, the four-step BPE record, an invalid head configuration, and untrusted input text.
- Every generated HTML view passed checks for unique IDs, control labels, valid direct lesson links, and existing local source downloads.
- 28 foreground/background palette pairs passed calculated contrast of at least 4.5:1 in light and dark themes. Minimum measured pair: 5.02:1. Focus-outline contrast was separately checked against the surface at 3:1. This is a mathematical token audit, not a rendered-page conformance audit.
- All 43 original source-package files still match their initial SHA-256 hashes. All 43 downloaded ZIP entries also match. No original lesson, Python lab, data, run, or checkpoint was edited.
- HTTP serving of the initial preview returned status 200. Static resources were served locally for final integrity checks.

## Learner journeys and their limits

| Journey | Actually exercised | Still needed |
|---|---|---|
| First visit → lesson → check → save → return | Core state transitions and save/restore logic; lesson/check templates | End-to-end clicks, focus order, real localStorage persistence across reload |
| Tokenizer round trip | Exact supplied IDs and decoded text in both modes, Unicode/whitespace edge cases | Typing and screen-reader announcements in the browser |
| Causal attention and shifted targets | Earlier allowed context/target invariance after future-ID edit; template generation | Manual step controls and keyboard focus in browser |
| Model shapes | Shape calculations, vocabulary changes, invalid heads | Select controls on desktop and mobile |
| Import and interpret a run | Parsing, fingerprint checks, identity/status consistency, malformed inputs, four-step BPE separation, unknown missing result | Native multi-file picker, browser readback, chart appearance |
| Notebook export/import | JSON serialization/validation/merge, preserving existing notes, storage failure | Native download and file-picker completion |
| Capstone evidence | Reference-required review validation; independent progress/evidence state; rubric templates | Evidence entry and export through the rendered UI |

## Accessibility implementation versus testing

Implemented: semantic landmarks and headings, explicit labels, fieldsets, native buttons/selects/disclosures, radio-group behavior, visible focus, main-region focus on route changes, mobile-menu focus and Escape, textual chart alternatives, non-color diagram labels, light/dark palettes, responsive rules, forced-colors support, and reduced-motion overrides. Simulations never autoplay.

Not tested in a browser: desktop/narrow rendered layout, touch targets in context, actual keyboard traversal and focus restoration, 200% zoom, OS reduced-motion response, screen-reader output, clipboard, downloads/file pickers, browser-storage quota behavior, or deployment runtime behavior. The WebMCP tools are feature-detected and implemented but could not be registered/invoked in a permitted supported browser context; their validation is unavailable.

## Scope boundaries

The Python training behavior is preserved byte-for-byte; its training tests were not rerun for this presentation-only change. Prior Python verification remains the supplied package's own record, not a new result. Tokenization runs in JavaScript using saved rules. Shape and causal views are explicit calculations/illustrations. Training curves and samples are supplied or imported records. There is no live Python training service, checkpoint loader, automated capstone grader, or cloud note synchronization.

## Retained evidence

- `qa/source-hashes.json`: baseline original-file identities.
- `qa/core.test.mjs`: reproducible functional cases.
- `qa/templates.mjs`: pure template checks (no browser runtime).
- `qa/audit.py` and `qa/accessibility-source.json`: generated markup, palette, and source/ZIP integrity checks.

## Tooling deviation

The Sites package's bundled publication helper became unavailable after Site registration; searching the installed plugin locations did not find it. The native connector remains the authority for any saved version and deployment. Any fallback source publisher must retain credentials only in memory, push an exact committed source tree without force, package only the static directory and manifest, and read back the remote commit before requesting a saved version. Browser policy restrictions remain in force.
