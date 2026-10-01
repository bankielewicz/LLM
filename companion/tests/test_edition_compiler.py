from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.compile_applied import (
    ANSWERS_FORMAT,
    APPLIED_IDS,
    CompilationError,
    LESSON_SECTIONS,
    MODULE_ORDER,
    OUTPUT_FORMAT,
    RUBRICS_FORMAT,
    SOURCE_FORMAT,
    compile_edition,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


class EditionFixture:
    def __init__(self, root: Path) -> None:
        self.repo = root
        self.source = root / "curriculum-applied" / "2026-09-28"
        self.spec = root / "docs" / "specs" / "active"
        self.output = root / "dist" / "applied-content.json"
        self.manifest_path = self.source / "manifest.json"
        self.answers_path = self.source / "answers.json"
        self.rubrics_path = self.source / "rubrics.json"
        self.overlay_path = self.spec / "fixtures" / "curriculum" / "curriculum-overlay.json"
        self.oracles_path = self.spec / "fixtures" / "curriculum" / "exercise-oracles.json"
        self.capstone_path = self.spec / "fixtures" / "curriculum" / "capstone-rubric.json"
        self.source_baseline_path = self.spec / "evidence" / "source-baseline.json"
        self._build()

    def _build(self) -> None:
        foundation_bytes = b'{"fixture":"foundation-content"}\n'
        (self.repo / "dist").mkdir(parents=True)
        (self.repo / "dist" / "content.json").write_bytes(foundation_bytes)
        foundation_digest = hashlib.sha256(foundation_bytes).hexdigest()

        foundation_modules = []
        for number in range(13):
            module_id = f"{number:02d}"
            relative = f"lessons/{module_id}.md"
            foundation_modules.append(
                {
                    "id": module_id,
                    "sequence": number,
                    "title": f"Foundation {module_id}",
                    "path": relative,
                    "outcome": f"Foundation outcome {module_id}",
                    "prerequisites": [],
                }
            )
            lesson = (
                f"# {module_id} — Foundation {module_id}\n\n"
                "## Bridge\n\n"
                "A stable bridge heading for supplement cards.\n"
            )
            path = self.repo / "course" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(lesson, encoding="utf-8")
        write_json(
            self.repo / "course" / "curriculum.json",
            {"schema_version": 1, "edition": "2026-09-26", "modules": foundation_modules},
        )
        baseline_commit = "3a47ea48da53cb9de3ff4727ca5f0f0b6f2b9bf8"
        protected_paths = [
            self.repo / "course" / "curriculum.json",
            self.repo / "dist" / "content.json",
            *[
                self.repo / "course" / module["path"]
                for module in foundation_modules
            ],
        ]
        write_json(
            self.source_baseline_path,
            {
                "source_commit": baseline_commit,
                "files_sha256": {
                    path.relative_to(self.repo).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in protected_paths
                },
            },
        )

        capstone_rubric = {
            "format": "llm-foundations-capstone-rubric-v1",
            "allowed_scores": [0, 5, 10],
            "rows": [{"id": "scope", "title": "Scope"}],
        }
        write_json(self.capstone_path, capstone_rubric)

        overlay_modules = []
        cards = []
        applied_manifest_modules = []
        answers = []
        rubrics = []
        oracles = []
        for position, module_id in enumerate(APPLIED_IDS):
            title = f"Applied {module_id}"
            overlay_modules.append(
                {
                    "id": module_id,
                    "title": title,
                    "outcome": f"Observable outcome for {module_id}",
                    "sequence": position,
                    "required": module_id in {str(value) for value in range(13, 21)},
                    "elective": module_id.startswith("E"),
                    "blocking_prerequisites": [],
                    "advisory_prerequisites": [],
                }
            )
            cards.append(
                {
                    "card_id": f"CARD-{module_id}",
                    "source_lesson_id": "00",
                    "anchor_heading": "## Bridge",
                    "anchor_occurrence": 1,
                    "order": position,
                    "kind": (
                        "prerequisite" if module_id == "P00"
                        else "elective" if module_id.startswith("E")
                        else "required"
                    ),
                    "destination_module_id": module_id,
                    "title": f"Open {module_id}",
                    "summary": f"Continue to {module_id}",
                    "estimated_minutes": 10,
                    "prerequisites": [],
                }
            )
            exercise_id = f"EX-{module_id}"
            answer_id = f"ANS-{module_id}"
            rubric_id = "RUBRIC-20" if module_id == "20" else None
            applied_manifest_modules.append(
                {
                    "id": module_id,
                    "source": f"lessons/{module_id}.md",
                    "exercises": [
                        {
                            "id": exercise_id,
                            "answer_id": answer_id,
                            "rubric_id": rubric_id,
                        }
                    ],
                }
            )
            answers.append(
                {
                    "id": answer_id,
                    "module_id": module_id,
                    "exercise_id": exercise_id,
                    "oracle_id": exercise_id,
                    "content": f"Bound reference answer for {module_id}.",
                }
            )
            oracles.append(
                {
                    "id": exercise_id,
                    "module_id": module_id,
                    "expected": {"accepted": True},
                }
            )
            if module_id == "20":
                rubrics.append(
                    {
                        "id": rubric_id,
                        "module_id": module_id,
                        "exercise_id": exercise_id,
                        "content": capstone_rubric,
                    }
                )
            sections = []
            for section_index, section in enumerate(LESSON_SECTIONS):
                marker = f"\n\n<!-- exercise:{exercise_id} -->" if section_index == 1 else ""
                sections.append(
                    f"## {section}\n\nConcrete source content for {module_id} and {section.lower()}.{marker}"
                )
            next_id = APPLIED_IDS[(position + 1) % len(APPLIED_IDS)]
            lesson = (
                f"# {module_id} — {title}\n\n"
                + "\n\n".join(sections)
                + f"\n\nContinue to [the next module](module:{next_id}#outcome-and-boundary).\n"
            )
            path = self.source / "lessons" / f"{module_id}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(lesson, encoding="utf-8")

        write_json(
            self.overlay_path,
            {
                "format": "llm-foundations-curriculum-overlay-v1",
                "edition_id": "intermediate-v1",
                "foundation_baseline_commit": baseline_commit,
                "completion_required": [str(value) for value in range(13, 21)],
                "modules": overlay_modules,
                "lesson_sections": list(LESSON_SECTIONS),
                "cards": cards,
            },
        )
        write_json(
            self.oracles_path,
            {"format": "llm-foundations-exercise-oracles-v1", "oracles": oracles},
        )
        write_json(
            self.manifest_path,
            {
                "format": SOURCE_FORMAT,
                "edition_id": "intermediate-v1",
                "foundation": {
                    "edition_id": "2026-09-26",
                    "content_sha256": foundation_digest,
                },
                "module_order": list(MODULE_ORDER),
                "applied_modules": applied_manifest_modules,
            },
        )
        write_json(self.answers_path, {"format": ANSWERS_FORMAT, "answers": answers})
        write_json(self.rubrics_path, {"format": RUBRICS_FORMAT, "rubrics": rubrics})
        fixture = self.source / "fixtures" / "teaching.json"
        write_json(fixture, {"id": "deterministic-teaching-fixture", "value": 7})
        write_json(self.source / "fixtures" / "Z-upper.json", {"case": "upper"})
        write_json(self.source / "fixtures" / "a-lower.json", {"case": "lower"})

    def compile(self, *, write: bool = False) -> dict[str, object]:
        return compile_edition(
            self.repo,
            self.source,
            self.output if write else None,
            spec_root=self.spec,
        )


class EditionCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = EditionFixture(Path(self.temporary.name))

    def test_compiles_complete_24_module_fixture_deterministically(self) -> None:
        first = self.fixture.compile(write=True)
        first_bytes = self.fixture.output.read_bytes()
        second = self.fixture.compile(write=True)

        self.assertEqual(first, second)
        self.assertEqual(first_bytes, self.fixture.output.read_bytes())
        self.assertEqual(json.loads(first_bytes), first)
        self.assertEqual(first["format"], OUTPUT_FORMAT)
        self.assertEqual(first["module_order"], list(MODULE_ORDER))
        self.assertEqual(len(first["modules"]), 24)
        self.assertEqual(
            [module["id"] for module in first["modules"]],
            list(MODULE_ORDER),
        )
        self.assertEqual(
            [item["path"] for item in first["teaching_fixtures"]],
            [
                "curriculum-applied/2026-09-28/fixtures/Z-upper.json",
                "curriculum-applied/2026-09-28/fixtures/a-lower.json",
                "curriculum-applied/2026-09-28/fixtures/teaching.json",
            ],
        )
        self.assertEqual(len({module["source_path"] for module in first["modules"]}), 24)
        for module in first["modules"]:
            source_bytes = (self.fixture.repo / module["source_path"]).read_bytes()
            self.assertEqual(module["source_sha256"], hashlib.sha256(source_bytes).hexdigest())
            self.assertTrue(module["body"])

        digest_input = {
            "edition_id": first["edition_id"],
            "foundation": first["foundation"],
            "module_order": first["module_order"],
            "source_files": first["source_files"],
        }
        canonical_digest_input = (
            json.dumps(
                digest_input,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        self.assertEqual(first["source_digest"], hashlib.sha256(canonical_digest_input).hexdigest())

    def test_failure_does_not_replace_preexisting_output(self) -> None:
        sentinel = b"previously-verified-output\n"
        self.fixture.output.write_bytes(sentinel)
        answers = read_json(self.fixture.answers_path)
        answers["answers"] = answers["answers"][1:]
        write_json(self.fixture.answers_path, answers)

        with self.assertRaisesRegex(CompilationError, "missing answer"):
            self.fixture.compile(write=True)
        self.assertEqual(self.fixture.output.read_bytes(), sentinel)

    def test_rejects_unknown_lesson_source(self) -> None:
        (self.fixture.source / "lessons" / "UNKNOWN.md").write_text(
            "# Unknown\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(CompilationError, "unknown source lessons"):
            self.fixture.compile()

    def test_rejects_unknown_and_duplicate_module_ids(self) -> None:
        manifest = read_json(self.fixture.manifest_path)
        manifest["applied_modules"][0]["id"] = "UNKNOWN"
        write_json(self.fixture.manifest_path, manifest)
        with self.assertRaisesRegex(CompilationError, "IDs/order"):
            self.fixture.compile()

        self.fixture = EditionFixture(Path(self.temporary.name) / "duplicate")
        manifest = read_json(self.fixture.manifest_path)
        manifest["applied_modules"].append(dict(manifest["applied_modules"][0]))
        write_json(self.fixture.manifest_path, manifest)
        with self.assertRaisesRegex(CompilationError, "duplicate id"):
            self.fixture.compile()

    def test_rejects_noncanonical_lesson_source_path(self) -> None:
        manifest = read_json(self.fixture.manifest_path)
        manifest["applied_modules"][0]["source"] = "lessons/13.md"
        write_json(self.fixture.manifest_path, manifest)
        with self.assertRaisesRegex(CompilationError, "expected 'lessons/P00.md'"):
            self.fixture.compile()

    def test_nonfinite_exponent_fails_before_output_replacement(self) -> None:
        text = self.fixture.answers_path.read_text(encoding="utf-8")
        self.fixture.answers_path.write_text(text.replace('"content": "Bound reference answer for P00."', '"content": 1e999'), encoding="utf-8")
        sentinel = b"retained-output\n"
        self.fixture.output.write_bytes(sentinel)
        with self.assertRaisesRegex(CompilationError, "non-finite number 1e999"):
            self.fixture.compile()
        with self.assertRaisesRegex(CompilationError, "non-finite number 1e999"):
            self.fixture.compile(write=True)
        self.assertEqual(self.fixture.output.read_bytes(), sentinel)

    def test_rejects_broken_module_link_and_anchor(self) -> None:
        lesson_path = self.fixture.source / "lessons" / "P00.md"
        lesson = lesson_path.read_text(encoding="utf-8")
        lesson_path.write_text(
            lesson.replace("module:13#outcome-and-boundary", "module:NOT-A-MODULE"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(CompilationError, "unknown destination"):
            self.fixture.compile()

        self.fixture = EditionFixture(Path(self.temporary.name) / "anchor")
        lesson_path = self.fixture.source / "lessons" / "P00.md"
        lesson = lesson_path.read_text(encoding="utf-8")
        lesson_path.write_text(
            lesson.replace("#outcome-and-boundary", "#missing-anchor"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(CompilationError, "unknown anchor"):
            self.fixture.compile()

    def test_rejects_unsupported_link_syntax_in_applied_sources(self) -> None:
        cases = {
            "reference": ("[reference][missing]", "reference-style links"),
            "autolink": ("<https://example.com>", "autolinks are unsupported"),
            "file-autolink": ("<file:missing.md>", "autolinks are unsupported"),
            "ftp-autolink": ("<ftp://example.test/x>", "autolinks are unsupported"),
            "html": ('<a href="missing.md">link</a>', "HTML href/src links"),
            "multiline-html": ('<a\n href="missing.md">link</a>', "HTML href/src links"),
            "nested": ("[nested](missing(file).md)", "parentheses in link destinations"),
        }
        for name, (injected, expected) in cases.items():
            with self.subTest(name=name):
                root = Path(self.temporary.name) / f"links-{name}"
                fixture = EditionFixture(root)
                lesson_path = fixture.source / "lessons" / "P00.md"
                lesson_path.write_text(
                    lesson_path.read_text(encoding="utf-8") + f"\n{injected}\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(CompilationError, expected):
                    fixture.compile()

    def test_foundation_reference_links_do_not_redefine_applied_source_syntax(self) -> None:
        foundation_path = self.fixture.repo / "course" / "lessons" / "00.md"
        foundation_path.write_text(
            foundation_path.read_text(encoding="utf-8") + "\n[historical][reference]\n",
            encoding="utf-8",
        )
        baseline = read_json(self.fixture.source_baseline_path)
        baseline["files_sha256"]["course/lessons/00.md"] = hashlib.sha256(
            foundation_path.read_bytes()
        ).hexdigest()
        write_json(self.fixture.source_baseline_path, baseline)
        self.fixture.compile()

    def test_rejects_answer_rubric_and_oracle_join_mismatches(self) -> None:
        answers = read_json(self.fixture.answers_path)
        answers["answers"][0]["module_id"] = "13"
        write_json(self.fixture.answers_path, answers)
        with self.assertRaisesRegex(CompilationError, "mismatched module/exercise join"):
            self.fixture.compile()

        self.fixture = EditionFixture(Path(self.temporary.name) / "rubric")
        rubrics = read_json(self.fixture.rubrics_path)
        rubrics["rubrics"][0]["exercise_id"] = "EX-19"
        write_json(self.fixture.rubrics_path, rubrics)
        with self.assertRaisesRegex(CompilationError, "mismatched module/exercise join"):
            self.fixture.compile()

        self.fixture = EditionFixture(Path(self.temporary.name) / "oracle")
        answers = read_json(self.fixture.answers_path)
        answers["answers"][0]["oracle_id"] = None
        write_json(self.fixture.answers_path, answers)
        with self.assertRaisesRegex(CompilationError, "bind this oracle exactly"):
            self.fixture.compile()

    def test_rejects_missing_sections_exercises_and_card_destinations(self) -> None:
        lesson_path = self.fixture.source / "lessons" / "13.md"
        lesson = lesson_path.read_text(encoding="utf-8")
        lesson_path.write_text(
            lesson.replace("## Feedback and limits", "### Feedback and limits"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(CompilationError, "level-2 sections"):
            self.fixture.compile()

        self.fixture = EditionFixture(Path(self.temporary.name) / "exercise")
        lesson_path = self.fixture.source / "lessons" / "13.md"
        lesson = lesson_path.read_text(encoding="utf-8")
        lesson_path.write_text(lesson.replace("<!-- exercise:EX-13 -->", ""), encoding="utf-8")
        with self.assertRaisesRegex(CompilationError, "exercise marker mismatch"):
            self.fixture.compile()

        self.fixture = EditionFixture(Path(self.temporary.name) / "card")
        overlay = read_json(self.fixture.overlay_path)
        overlay["cards"][0]["destination_module_id"] = "UNKNOWN"
        write_json(self.fixture.overlay_path, overlay)
        with self.assertRaisesRegex(CompilationError, "unknown destination module"):
            self.fixture.compile()



    def test_rejects_comment_and_markup_only_placeholders(self) -> None:
        cases = (("<!-- TODO -->", "empty"), ("**TODO**", "placeholder"))
        for position, (replacement, expected) in enumerate(cases):
            with self.subTest(replacement=replacement):
                fixture = EditionFixture(Path(self.temporary.name) / f"placeholder-{position}")
                lesson_path = fixture.source / "lessons" / "13.md"
                lesson = lesson_path.read_text(encoding="utf-8")
                lesson_path.write_text(
                    lesson.replace(
                        "Concrete source content for 13 and feedback and limits.",
                        replacement,
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(CompilationError, expected):
                    fixture.compile()

    def test_card_contract_is_closed_and_unambiguous(self) -> None:
        mutations = {
            "unknown-field": (
                lambda card: card.__setitem__("unexpected", True),
                "unknown fields",
            ),
            "occurrence": (
                lambda card: card.__setitem__("anchor_occurrence", 2),
                "expected exactly 1",
            ),
            "kind": (
                lambda card: card.__setitem__("kind", "optional"),
                "expected prerequisite, required, or elective",
            ),
            "prerequisite": (
                lambda card: card.__setitem__("prerequisites", ["UNKNOWN"]),
                "unknown prerequisite module",
            ),
            "duplicate-prerequisite": (
                lambda card: card.__setitem__("prerequisites", ["00", "00"]),
                "duplicate prerequisite module",
            ),
        }
        for name, (mutate, expected) in mutations.items():
            with self.subTest(name=name):
                fixture = EditionFixture(Path(self.temporary.name) / f"card-{name}")
                overlay = read_json(fixture.overlay_path)
                mutate(overlay["cards"][0])
                write_json(fixture.overlay_path, overlay)
                with self.assertRaisesRegex(CompilationError, expected):
                    fixture.compile()

        fixture = EditionFixture(Path(self.temporary.name) / "card-ambiguous")
        lesson_path = fixture.repo / "course" / "lessons" / "00.md"
        lesson_path.write_text(
            lesson_path.read_text(encoding="utf-8") + "\n## Bridge\n\nSecond heading.\n",
            encoding="utf-8",
        )
        baseline = read_json(fixture.source_baseline_path)
        baseline["files_sha256"]["course/lessons/00.md"] = hashlib.sha256(
            lesson_path.read_bytes()
        ).hexdigest()
        write_json(fixture.source_baseline_path, baseline)
        with self.assertRaisesRegex(CompilationError, "absent or ambiguous"):
            fixture.compile()

    def test_output_is_fixed_and_rejects_symlinks_before_writing(self) -> None:
        protected = self.fixture.repo / "course" / "lessons" / "00.md"
        protected_bytes = protected.read_bytes()
        with self.assertRaisesRegex(CompilationError, "exactly dist/applied-content.json"):
            compile_edition(
                self.fixture.repo,
                self.fixture.source,
                protected,
                spec_root=self.fixture.spec,
            )
        self.assertEqual(protected.read_bytes(), protected_bytes)

        link_target = self.fixture.repo / "course" / "link-target.txt"
        link_target.write_bytes(b"protected-link-target\n")
        self.fixture.output.symlink_to(link_target)
        with self.assertRaisesRegex(CompilationError, "symlink or junction"):
            self.fixture.compile(write=True)
        self.assertEqual(link_target.read_bytes(), b"protected-link-target\n")

        fixture = EditionFixture(Path(self.temporary.name) / "ancestor-link")
        actual_dist = fixture.repo / "actual-dist"
        fixture.repo.joinpath("dist").rename(actual_dist)
        fixture.repo.joinpath("dist").symlink_to(actual_dist, target_is_directory=True)
        target = actual_dist / "applied-content.json"
        target.write_bytes(b"protected-ancestor-target\n")
        with self.assertRaisesRegex(CompilationError, "symlink or junction"):
            fixture.compile(write=True)
        self.assertEqual(target.read_bytes(), b"protected-ancestor-target\n")

    def test_rejects_symlinked_teaching_fixture_root(self) -> None:
        fixtures_dir = self.fixture.source / "fixtures"
        linked_fixtures = self.fixture.repo / "linked-fixtures"
        fixtures_dir.rename(linked_fixtures)
        fixtures_dir.symlink_to(linked_fixtures, target_is_directory=True)

        with self.assertRaisesRegex(CompilationError, "fixture root may not be a symlink"):
            self.fixture.compile()

    def test_rejects_authority_format_and_baseline_mismatches(self) -> None:
        overlay = read_json(self.fixture.overlay_path)
        overlay["format"] = "wrong-overlay-format"
        write_json(self.fixture.overlay_path, overlay)
        with self.assertRaisesRegex(CompilationError, "curriculum overlay.format"):
            self.fixture.compile()

        fixture = EditionFixture(Path(self.temporary.name) / "oracle-format")
        oracles = read_json(fixture.oracles_path)
        oracles["format"] = "wrong-oracle-format"
        write_json(fixture.oracles_path, oracles)
        with self.assertRaisesRegex(CompilationError, "exercise oracles.format"):
            fixture.compile()

        fixture = EditionFixture(Path(self.temporary.name) / "baseline")
        overlay = read_json(fixture.overlay_path)
        overlay["foundation_baseline_commit"] = "0" * 40
        write_json(fixture.overlay_path, overlay)
        with self.assertRaisesRegex(CompilationError, "does not match source baseline"):
            fixture.compile()
if __name__ == "__main__":
    unittest.main()
