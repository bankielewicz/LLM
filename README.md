# LLM Foundations learner interface

A static learning application built around the supplied 2026-09-26 course. The original sibling `llm-foundations-v2` package is untouched. Its 43 source files are copied byte-for-byte into `course/` and the downloadable lab ZIP.

## Open locally

From this folder, run `python3 -m http.server 4173 --directory dist` in WSL/Linux, or `python -m http.server 4173 --directory dist` in Windows. Open http://localhost:4173 in your browser. Keep that terminal running. No Node packages or Python dependencies are needed for the interface.

The Python training labs have their own PyTorch requirements and must be run from the downloaded `llm-foundations-v2` folder. The interface does not run training.

## Content and presentation

- `course/` retains the supplied Markdown, curriculum, examples, references, and Python labs.
- `dist/content.json` supplies all 13 canonical lessons and recorded examples to the reader.
- `dist/core.js` implements safe Markdown rendering, UTF-8/BPE representation, shape calculation, and validated local records.
- `dist/app.js` contains page templates and interactions; `dist/styles.css` defines both themes and responsive behavior.
- `scripts/sync_course.py` refreshes content and the lab ZIP from the sibling course. Use UTF-8 Python (`python -X utf8 scripts/sync_course.py` on Windows). The initial source-hash receipt prevents unnoticed source drift; review source changes before replacing that receipt for a future edition.
- Hash routes support direct lesson links such as `/#/lesson/05` on a basic static host.

## Learner data

Notes, reading/practice/check state, evidence references, and imported runs use localStorage under `llm-foundations-v1`. They are local to one browser profile and origin. A localhost preview and a hosted Site do not share storage. JSON export/import transfers them; import merges progress and appends differing notes. Text export is also available. Storage failure leaves the current session usable with an explicit export reminder.

Imported run files must be the named `manifest.json`, `metrics.jsonl`, optional `result.json`, and optional `sample.txt`, under 1 MB combined. At most 20 imports are kept. The importer checks record structure, configuration, tokenizer fingerprint, monotonically increasing steps, finite measurements, and result consistency. It cannot verify that the original Python process ran or independently inspect checkpoints/data. Missing status is unknown.

## Verification

Run `node --test qa/core.test.mjs`, `node qa/templates.mjs`, and `python -X utf8 qa/audit.py`. See `QA_REPORT.md` for tested journeys and the explicit browser-testing limitation. The tests use no browser and require no external packages.
