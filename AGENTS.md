# Repository Guidelines

## Source Layout

```text
companion/
  src/llm_foundations_companion/
    *.py                    # API, auth, storage, scheduling, workers
    operations/             # operation handlers
    contract_data/          # bundled contracts and schemas
    migrations/             # database upgrades
    static/                 # companion browser assets
  tests/                    # pytest suites
  locks/                    # pinned dependency profiles
dist/                       # deployed static learner site
course/                     # preserved lessons and Python labs
qa/                         # Node regressions and source audits
scripts/                    # compilers, custody checks, release gates
docs/                       # specifications and implementation records
```

Canonical course: `/home/bryan/Projects/llm-foundations-v2`. Regenerate copies and bundles only through their owning generators; preserve source receipts.

## Build, Test, and Development Commands

Run from the worktree root with Python 3.12; prepare development/runtime environments from `companion/locks/README.md`.

```bash
python3 -m http.server 4173 --directory dist # static preview
node --test qa/core.test.mjs qa/import-regressions.test.mjs
python -m coverage run --source=companion/src/llm_foundations_companion -m pytest companion/tests
python -m coverage report --fail-under=95
python -B scripts/run_s1_gate.py --help # complete service/regression gate
```

Use the slice gates for release checks. Run the legacy template/hash audit in a disposable copy because it writes a protected report.

## TDD and Coverage

Follow red–green–refactor: test each behavior change first, verify the intended failure, implement minimally, then refactor with tests green. Companion tests use `pytest` and `test_*.py`; JavaScript uses `node:test` and `*.test.mjs`; legacy labs use `unittest`.

Require **at least 95% line coverage** for each affected executable component, measuring its complete source tree. Include reproducible coverage commands/results in PRs. Never exclude untested production logic, lower the threshold, or count skips/`NOT_RUN` as passing. Documentation-only changes require applicable documentation and custody checks.

## Coding Style and Preservation

Use UTF-8, LF, and `core.autocrlf=false`. Use four-space Python indentation, snake_case, JavaScript camelCase, and existing formatting. Escape learner text with `C.escapeHTML`. Preserve approved specifications and protected baseline bytes; amendments require separate authorization.

## Worktrees, Commits, and Pull Requests

Before modifying any repository file, create a dedicated `codex/<topic>` branch and Git worktree from current `origin/main`; keep the primary checkout untouched. Use imperative commit subjects, e.g. `docs: require TDD and coverage`.

Complete all applicable tests, coverage gates, and validation before pushing. Failures or missing required checks block publication. Once checks pass, push the task branch and open a PR targeting `main`. PRs include scope, linked requirements/issues, test/coverage results, limitations, and UI screenshots. Leave merging to the owner.

## Tooling Obstacles

Consult `C:/Users/bryan/.codex/code/papercuts.md` and `C:/Users/bryan/.claude/code/papercuts.md` when tooling fails. Verify recorded fixes. Append dated symptom, fix/status, and project only to the Codex log; preserve entries, avoid duplicates, and exclude secrets.
