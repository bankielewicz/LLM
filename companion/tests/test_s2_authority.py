"""Mutation controls for the external APP-009 authority overlay."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_s2_authority as authority


class S2AuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.scratch = Path(self.temp.name)

    def fixture(self) -> Path:
        repo = self.scratch / "repo"
        source_spec = ROOT / authority.SPEC_ROOT_REL
        manifest = json.loads((source_spec / "spec-manifest.json").read_text())
        for row in manifest["files"]:
            relative = Path(row["path"])
            target = repo / authority.SPEC_ROOT_REL / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_spec / relative, target)

        extras = (
            authority.SPEC_MANIFEST_REL,
            authority.S0_AUTHORITY_REL,
            authority.S2_BASELINE_REL,
            authority.PROPOSAL_REL,
            authority.AMENDMENT_REL,
            authority.SIDECAR_REL,
        )
        for relative in extras:
            source = ROOT / relative
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return repo

    @staticmethod
    def rewrite_json(path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def test_live_overlay_preserves_the_sealed_base_and_denominator(self) -> None:
        result = authority.audit(ROOT)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["effective_revision"], "1.3")
        self.assertEqual(result["base"]["revision"], "1.2")
        self.assertEqual(result["base"]["file_count"], 150)
        self.assertEqual(
            result["base"]["payload_sha256"],
            authority.EXPECTED_SPEC_PAYLOAD_SHA256,
        )
        self.assertEqual(
            result["acceptance"],
            {"cases": 128, "execution_units": 605, "status": "NOT_RUN"},
        )
        target = (ROOT / authority.TARGET_REL).read_text(encoding="utf-8")
        self.assertEqual(target.count(authority.OLD_PHRASE), 1)
        self.assertEqual(target.count(authority.NEW_PHRASE), 0)

    def test_editing_the_sealed_app009_phrase_is_rejected(self) -> None:
        repo = self.fixture()
        target = repo / authority.TARGET_REL
        text = target.read_text(encoding="utf-8")
        self.assertEqual(text.count(authority.OLD_PHRASE), 1)
        target.write_text(
            text.replace(authority.OLD_PHRASE, authority.NEW_PHRASE),
            encoding="utf-8",
            newline="\n",
        )
        with self.assertRaisesRegex(
            authority.AuthorityError,
            "base manifest member differs: 06-APPLIED-MODELS.md",
        ):
            authority.audit(repo)

    def test_changing_the_frozen_acceptance_denominator_is_rejected(self) -> None:
        repo = self.fixture()
        path = repo / authority.ACCEPTANCE_REL
        value = json.loads(path.read_text(encoding="utf-8"))
        value["execution_units"].pop()
        self.rewrite_json(path, value)
        with self.assertRaisesRegex(
            authority.AuthorityError,
            "base manifest member differs: acceptance-cases.json",
        ):
            authority.audit(repo)

    def test_proposal_and_amendment_bytes_are_independently_bound(self) -> None:
        for relative, message in (
            (authority.PROPOSAL_REL, "approved proposal SHA-256 differs"),
            (authority.AMENDMENT_REL, "normative amendment SHA-256 differs"),
        ):
            with self.subTest(relative=relative):
                repo = self.fixture()
                path = repo / relative
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaisesRegex(authority.AuthorityError, message):
                    authority.audit(repo)
                shutil.rmtree(repo)

    def test_sidecar_is_closed_and_cannot_repoint_the_amendment(self) -> None:
        mutations = (
            (
                lambda value: value.update({"unexpected": True}),
                "sidecar fields differ",
            ),
            (
                lambda value: value["amendment"].update(
                    {"path": "../outside-amendment.md"}
                ),
                "amendment path differs",
            ),
            (
                lambda value: value["preserved_contracts"].update(
                    {"acceptance_execution_unit_count": 604}
                ),
                "preserved contracts acceptance_execution_unit_count differs",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                repo = self.fixture()
                path = repo / authority.SIDECAR_REL
                value = json.loads(path.read_text(encoding="utf-8"))
                mutate(value)
                self.rewrite_json(path, value)
                with self.assertRaisesRegex(authority.AuthorityError, message):
                    authority.audit(repo)
                shutil.rmtree(repo)

    def test_semantically_identical_sidecar_rewrite_is_rejected(self) -> None:
        repo = self.fixture()
        path = repo / authority.SIDECAR_REL
        original = path.read_bytes()
        value = json.loads(original)
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        self.assertNotEqual(path.read_bytes(), original)
        with self.assertRaisesRegex(
            authority.AuthorityError,
            "authority sidecar SHA-256 differs",
        ):
            authority.audit(repo)

    def test_symlink_cannot_substitute_for_the_bound_amendment(self) -> None:
        repo = self.fixture()
        amendment = repo / authority.AMENDMENT_REL
        external = self.scratch / "external-amendment.md"
        shutil.copy2(amendment, external)
        amendment.unlink()
        try:
            os.symlink(external, amendment)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        with self.assertRaisesRegex(
            authority.AuthorityError,
            "normative amendment path crosses a symlink",
        ):
            authority.audit(repo)


if __name__ == "__main__":
    unittest.main()
