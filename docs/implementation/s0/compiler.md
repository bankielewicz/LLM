# S0 applied-edition compiler

Status: implemented compiler contract and synthetic fixture tests. This does not report that the applied curriculum has been authored or that `dist/applied-content.json` has been produced for a release.

## Purpose and authority

`scripts/compile_applied.py` builds the intermediate content overlay without changing the protected foundation course. It joins three authorities:

1. `course/curriculum.json`, its 00–12 Markdown sources, the byte identity of `dist/content.json`, and the active specification's `evidence/source-baseline.json` receipt;
2. the active specification's curriculum overlay, exercise oracles, and capstone rubric; and
3. the applied source edition under `curriculum-applied/2026-09-28/`.

The active specification is selected by the repository-relative `spec_root` in `docs/implementation/s0/authority.json`. Tests and specialized callers may pass `spec_root` explicitly. The compiler does not hard-code a specification revision directory.

The generated module order is exactly `P00`, `00` through `20`, `E01`, `E02`. Foundation bodies are read from their protected source paths. Applied bodies come only from the edition manifest. The compiler rejects incomplete joins before publishing output.

## Source layout

The applied source directory has this contract:

```text
curriculum-applied/2026-09-28/
  README.md
  manifest.json
  answers.json
  rubrics.json
  lessons/
    P00.md
    13.md
    ...
    20.md
    E01.md
    E02.md
  fixtures/
    ... authored teaching inputs ...
```

The `README.md` is author guidance and is not a compiled source. Every regular file beneath `fixtures/` is included in the aggregate source digest. Symlinked teaching fixtures are rejected. The set of Markdown files beneath `lessons/` must equal the set named by the manifest: an undeclared file is an unknown source, and a declared file that is absent is missing source content.

JSON inputs use strict UTF-8. Duplicate object keys, non-finite numbers (including finite-looking exponent syntax that overflows), unknown fields in compiler-owned source records, and non-normalized or escaping paths fail compilation.

## Manifest

`manifest.json` has exactly these top-level fields:

| Field | Meaning |
|---|---|
| `format` | `llm-foundations-applied-source-v1` |
| `edition_id` | `intermediate-v1` |
| `foundation` | Object with `edition_id` and lowercase 64-character `content_sha256` |
| `module_order` | Exact 24-ID order described above |
| `applied_modules` | Exact order `P00`, `13`–`20`, `E01`, `E02` |

`foundation.edition_id` must equal `course/curriculum.json.edition`. `foundation.content_sha256` is the SHA-256 of the raw protected `dist/content.json` bytes and must match at compile time.

Each `applied_modules` item has exactly:

```json
{
  "id": "13",
  "source": "lessons/13.md",
  "exercises": [
    {
      "id": "M13-LEAKY",
      "answer_id": "ANSWER-M13-LEAKY",
      "rubric_id": null
    }
  ]
}
```

The source path is fixed by module ID: each applied module `<ID>` must name exactly `lessons/<ID>.md`.

Module IDs and exercise IDs are globally unique. Every exercise names an answer, a rubric, or both. A lesson contains exactly one marker for each declared exercise:

```markdown
<!-- exercise:M13-LEAKY -->
```

Unknown, missing, or duplicate markers fail compilation.

## Answers, rubrics, and normative oracles

`answers.json` has format `llm-foundations-applied-answers-v1` and an `answers` array. Each answer has exactly `id`, `module_id`, `exercise_id`, `oracle_id`, and nonempty Markdown `content`. `oracle_id` is either null or an exact ID from the active specification's `exercise-oracles.json`.

`rubrics.json` has format `llm-foundations-applied-rubrics-v1` and a `rubrics` array. Each rubric has exactly `id`, `module_id`, `exercise_id`, and a nonempty object in `content`.

The compiler enforces all joins in both directions:

- every answer and rubric is bound exactly once;
- the bound module and exercise IDs agree at every side of the join;
- every normative exercise oracle has a compiled exercise with the same module ID and an answer whose `oracle_id` names it;
- an answer cannot name an unknown oracle; and
- one rubric bound to module 20 must equal the active specification's complete capstone rubric object.

These checks establish structural identity and exact machine-readable oracle custody. They do not determine whether free-form explanatory prose is pedagogically or factually sufficient; content review remains separate.

## Applied Markdown

Each applied lesson uses one level-1 heading with the exact text `<module ID> — <overlay title>`. It then uses these level-2 headings, once each and in this order:

1. Outcome and boundary
2. Predict before running
3. Guided worked example
4. Partially scaffolded practice
5. Independent evidence task
6. Feedback and limits
7. Stopping point

Every section must contain visible text. A section containing only `TODO`, `TBD`, `placeholder`, `coming soon`, or `under construction` is rejected. Headings, links, and exercise markers inside fenced code blocks are treated as examples rather than document structure.

Heading anchors are deterministic: headings are NFKC-normalized and case-folded; letters, numbers, hyphens, and underscores remain; whitespace becomes a hyphen; other characters are removed; repeated anchors receive `-1`, `-2`, and so on. Authors may use an explicit final `{#anchor}` on a heading.

Applied lessons use Markdown inline links only. Reference-style links, autolinks, HTML `href`/`src` links, malformed inline links, and unescaped parentheses in link destinations fail compilation. Inline-link targets are checked as follows:

- `#anchor` must name an anchor in the current lesson;
- `module:13#anchor` must name a compiled module and, when supplied, an anchor in that module;
- relative file links must resolve to regular files inside the repository, and Markdown fragments must exist; and
- `http`, `https`, and `mailto` are permitted but are not fetched during compilation.

Other URI schemes and paths that escape the repository fail compilation. Existing foundation reference-style syntax is not redefined by the applied-source grammar; inline links found in protected foundation bodies are still target-validated.

The active overlay's supplement cards are closed records with exactly `card_id`, `source_lesson_id`, `anchor_heading`, `anchor_occurrence`, `order`, `kind`, `destination_module_id`, `title`, `summary`, `estimated_minutes`, and `prerequisites`. Card IDs and insertion order keys are unique. `anchor_occurrence` is exactly 1, and the named foundation heading must occur exactly once. Card kind must be `prerequisite` for P00, `required` for 13–20, and `elective` for E01/E02. Prerequisites must be unique known module IDs. The destination set covers every applied module exactly once.

The overlay must have the expected v1 format, exact lesson grammar, and completion-required IDs 13–20. Its lowercase 40-character `foundation_baseline_commit` must equal `source-baseline.json.source_commit`; every protected foundation curriculum, lesson, and compiled-content hash must equal the receipt.

## Output and digests

The output format is `llm-foundations-applied-content-v1`. It contains:

- `edition_id`, foundation identity, and the exact module order;
- all 24 compiled modules with `source_kind`, repository-relative `source_path`, raw `source_sha256`, body, and compiled exercise joins;
- completion metadata and supplement cards from the active overlay;
- repository-relative `source_files` with raw SHA-256 values; and
- hashed teaching-fixture inventory.

`source_digest` is the SHA-256 of this canonical object:

```json
{
  "edition_id": "intermediate-v1",
  "foundation": {
    "edition_id": "2026-09-26",
    "content_sha256": "the verified foundation content digest",
    "baseline_commit": "the verified foundation source commit"
  },
  "module_order": ["the exact 24 module IDs"],
  "source_files": ["the sorted path and SHA-256 records"]
}
```

Canonical JSON uses UTF-8, sorted object keys, no insignificant whitespace, no ASCII escaping, no non-finite numbers, and one final LF. `source_files` includes the foundation curriculum and lessons, protected `dist/content.json`, the active source-baseline receipt, active overlay/oracles/capstone rubric, applied manifest/answers/rubrics/lessons, and every teaching fixture. Repository-relative paths make the digest stable across checkout locations while a change of active authority remains visible.

Teaching fixtures are ordered explicitly by repository-relative POSIX path. The complete output uses the same canonical JSON encoding. Per-module hashes cover raw Markdown bytes. The aggregate digest is therefore explicit and independently reproducible; it is not a claim that the content has passed browser, runtime, or learner acceptance.

## API and command line

The Python API is:

```python
compile_edition(repo_root, source_root, output_path=None, *, spec_root=None) -> dict
```

Omitting `output_path` performs a read-only validation and returns the compiled object. The only accepted publication target is the exact lexical path `<repo_root>/dist/applied-content.json`. The compiler rejects a symlink or junction at `dist` or the output file before it creates a directory or writes bytes. A valid publication uses a same-directory temporary file, flush, `fsync`, and atomic replace after all validation passes. A failed compile does not create or replace the output.

From the repository root, the normal commands are:

```text
python3 scripts/compile_applied.py
python3 scripts/compile_applied.py --check
```

The first command reads the default edition source and publishes `dist/applied-content.json`. The second recompiles in memory and requires the existing output bytes to be exact. `--repo-root`, `--source-root`, and `--spec-root` provide explicit inputs for tests and controlled builds. `--output` is accepted only when it names the fixed publication path.

## S0 verification boundary

`companion/tests/test_edition_compiler.py` constructs a complete synthetic 24-module source tree and checks deterministic output, all module source hashes, the aggregate digest, and explicit case-sensitive fixture ordering. Mutation cases cover unknown and duplicate IDs; noncanonical lesson paths; non-finite exponents; unknown lesson files; missing answers; answer/rubric/oracle join mismatches; missing sections and exercise markers; comment- or markup-only placeholders; unsupported applied link forms; broken module destinations and anchors; closed supplement-card records, kinds, prerequisites, and unambiguous insertion headings; authority formats; and baseline receipt mismatches. Publication cases seed existing output, protected, and symlink targets and verify that their bytes remain unchanged.

The fixture proves the compiler behavior without shipping fabricated curriculum. Authoring the real applied lessons, answers, rubrics, and teaching fixtures remains required before the default source directory can compile.
