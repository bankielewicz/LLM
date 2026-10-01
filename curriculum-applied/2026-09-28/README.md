# Intermediate v1 curriculum source

This directory is reserved for the authored intermediate overlay. S0 supplies the compiler and its source contract; it does not fill this directory with lesson stubs or claim the curriculum is complete.

Before adding content, read [the compiler contract](../../docs/implementation/s0/compiler.md) and the active specification named by `docs/implementation/s0/authority.json`.

A complete source edition contains:

- `manifest.json`, `answers.json`, and `rubrics.json` using the documented v1 formats;
- exactly eleven applied lesson files under `lessons/`: P00, 13–20, E01, and E02;
- each module source fixed to `lessons/<module ID>.md` and each lesson using Markdown inline links only;
- every active-spec exercise oracle joined to a compiled exercise and reference answer;
- the exact active-spec capstone rubric bound to a module-20 exercise; and
- any authored deterministic teaching inputs under `fixtures/`.

Do not copy or rewrite lessons 00–12 here. The compiler reads their protected bodies from `course/curriculum.json` and verifies them against the active specification's source-baseline receipt. Do not put draft Markdown in `lessons/`; every discovered lesson is treated as release source and must be declared, complete, linked, and joined to its answer/rubric records.

Reference-style links, autolinks, HTML links, and unescaped parentheses in link destinations are outside the applied-source grammar. Use `[visible text](target)` and percent-encode parentheses in targets.

Run the focused compiler tests while authoring:

```text
python3 -m unittest companion.tests.test_edition_compiler -v
```

Once all real sources exist, compile the edition and then verify byte stability:

```text
python3 scripts/compile_applied.py
python3 scripts/compile_applied.py --check
```

A successful compile establishes deterministic source/output correspondence. It does not establish browser behavior, companion execution, product qualification, or learner outcomes.
