"""Independent failure controls for frozen source and manifest custody."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.util.spec_from_file_location("check_custody", ROOT / "scripts/check_custody.py")
custody = importlib.util.module_from_spec(loader)
loader.loader.exec_module(custody)


class CustodyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.spec = self.repo / "docs/specs/test"
        self.spec.mkdir(parents=True)
        (self.spec / "README.md").write_bytes(b"Frozen specification\n")
        self.source = self.repo / "course/lesson.md"
        self.source.parent.mkdir()
        self.source.write_bytes(b"Protected foundation\n")
        self.inventory = self.spec / "source-baseline.json"
        self.inventory.write_text(json.dumps({"files_sha256": {"course/lesson.md": custody.sha256(self.source.read_bytes())}}), encoding="utf-8")
        rows = custody.manifest_files(self.spec)
        self.manifest = self.spec / "spec-manifest.json"
        self.manifest.write_text(json.dumps({"spec_revision": "1.1", "exclusions": custody.EXCLUSIONS,
                                            "files": rows, "payload_sha256": custody.sha256(custody.canonical(rows))}), encoding="utf-8")
        self.expected = {"revision": "1.1", "file_count": len(rows),
                         "manifest_sha256": custody.sha256(self.manifest.read_bytes()),
                         "payload_sha256": custody.sha256(custody.canonical(rows))}
        self.protected = {"file_count": 1, "inventory_sha256": custody.sha256(self.inventory.read_bytes())}

    def test_unchanged_bytes_pass(self):
        self.assertEqual(custody.verify_spec(self.spec, self.expected)["files"], 2)
        self.assertEqual(custody.verify_protected(self.repo, self.inventory, self.protected)["files"], 1)

    def test_lf_to_crlf_change_is_rejected(self):
        self.source.write_bytes(b"Protected foundation\r\n")
        with self.assertRaisesRegex(custody.CustodyError, "Protected source changed"):
            custody.verify_protected(self.repo, self.inventory, self.protected)

    def test_missing_protected_file_is_rejected(self):
        self.source.unlink()
        with self.assertRaisesRegex(custody.CustodyError, "missing"):
            custody.verify_protected(self.repo, self.inventory, self.protected)

    def test_rewritten_receipt_cannot_rebind_changed_source(self):
        self.source.write_bytes(b"Changed source\n")
        self.inventory.write_text(json.dumps({"files_sha256": {"course/lesson.md": custody.sha256(self.source.read_bytes())}}), encoding="utf-8")
        with self.assertRaisesRegex(custody.CustodyError, "replacement receipt"):
            custody.verify_protected(self.repo, self.inventory, self.protected)

    def test_unlisted_normative_file_is_rejected(self):
        (self.spec / "unsealed.json").write_bytes(b"{}\n")
        with self.assertRaisesRegex(custody.CustodyError, "unsealed.json"):
            custody.verify_spec(self.spec, self.expected)

    def test_changed_normative_file_is_rejected(self):
        (self.spec / "README.md").write_bytes(b"Changed specification\n")
        with self.assertRaisesRegex(custody.CustodyError, "README.md"):
            custody.verify_spec(self.spec, self.expected)

    def test_resealed_changes_cannot_rebind_authority(self):
        (self.spec / "README.md").write_bytes(b"Changed specification\n")
        value = json.loads(self.manifest.read_text())
        value["files"] = custody.manifest_files(self.spec)
        value["payload_sha256"] = custody.sha256(custody.canonical(value["files"]))
        self.manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(custody.CustodyError, "manifest identity"):
            custody.verify_spec(self.spec, self.expected)

    def test_new_review_report_is_outside_seal(self):
        (self.spec / "reviews").mkdir()
        (self.spec / "reviews/new.json").write_bytes(b"{}\n")
        self.assertEqual(custody.verify_spec(self.spec, self.expected)["files"], 2)

    def test_path_traversal_and_windows_ambiguity_are_rejected(self):
        for value in ("../secret", "/tmp/secret", "course/../secret", "course//lesson.md",
                      r"course\lesson.md", "C:/secret", "course/./lesson.md"):
            with self.subTest(value=value), self.assertRaises(custody.CustodyError):
                custody.contained(self.repo, value)

    def test_symlink_cannot_replace_protected_file(self):
        other = self.repo / "outside.md"
        other.write_bytes(self.source.read_bytes())
        self.source.unlink()
        try:
            self.source.symlink_to(other)
        except (OSError, NotImplementedError):
            self.skipTest("Host does not permit symlink creation")
        with self.assertRaisesRegex(custody.CustodyError, "Symlink"):
            custody.verify_protected(self.repo, self.inventory, self.protected)

    def test_duplicate_json_keys_are_rejected(self):
        (self.repo / "duplicate.json").write_bytes(b'{"files":1,"files":2}')
        with self.assertRaisesRegex(custody.CustodyError, "Duplicate JSON key"):
            custody.strict_json(self.repo / "duplicate.json")


class ApprovedHistoryTests(unittest.TestCase):
    def setUp(self):
        import subprocess
        self.subprocess = subprocess
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.spec = self.repo / "docs/specs/intermediate-v1"
        self.spec.mkdir(parents=True)
        (self.spec / "evidence").mkdir()
        self.source = self.repo / "course/lesson.md"
        self.source.parent.mkdir()
        self.source.write_bytes(b"Approved source\n")
        (self.spec / "README.md").write_bytes(b"Approved specification\n")
        self.inventory = self.spec / "evidence/source-baseline.json"
        self.inventory.write_text(json.dumps({"files_sha256": {"course/lesson.md": custody.sha256(self.source.read_bytes())}}))
        self.make_manifest("1.1")
        self.git("init", "-b", "source")
        self.git("add", ".")
        self.git("commit", "-m", "Synthetic approved authority")
        commit = self.git("rev-parse", "HEAD").strip()
        self.retained = self.repo / "docs/implementation/s0/approved-1.1-manifest.json"
        self.retained.parent.mkdir(parents=True)
        self.retained.write_bytes((self.spec / "spec-manifest.json").read_bytes())
        self.approved = self.expected("1.1")
        self.approved.update(commit=commit, retained_manifest=self.retained.relative_to(self.repo).as_posix())
        (self.spec / "README.md").write_bytes(b"Explicitly amended specification\n")
        self.make_manifest("1.2")
        self.authority = self.retained.parent / "authority.json"
        self.binding = {
            "format": "llm-foundations-s0-authority-v1", "spec_root": "docs/specs/intermediate-v1",
            "approved_baseline": self.approved, "active_specification": self.expected("1.2"),
            "protected_source": {"inventory": self.inventory.relative_to(self.repo).as_posix(),
                                 "inventory_sha256": custody.sha256(self.inventory.read_bytes()), "file_count": 1},
        }
        self.write_binding()

    def git(self, *args):
        return self.subprocess.run(
            ["git", "-c", "user.name=S0 test", "-c", "user.email=s0-test@example.invalid",
             "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false",
             "-c", "core.hooksPath=" + str(self.repo / "empty-hooks"), "-C", str(self.repo), *args],
            check=True, capture_output=True, text=True).stdout

    def make_manifest(self, revision):
        rows = custody.manifest_files(self.spec)
        (self.spec / "spec-manifest.json").write_text(json.dumps({
            "spec_revision": revision, "exclusions": custody.EXCLUSIONS, "files": rows,
            "payload_sha256": custody.sha256(custody.canonical(rows))}))

    def expected(self, revision):
        path = self.spec / "spec-manifest.json"
        manifest = json.loads(path.read_text())
        return {"spec_root": "docs/specs/intermediate-v1", "revision": revision,
                "file_count": len(manifest["files"]), "manifest_sha256": custody.sha256(path.read_bytes()),
                "payload_sha256": manifest["payload_sha256"]}

    def write_binding(self):
        self.authority.write_text(json.dumps(self.binding))

    def test_approved_history_and_new_revision_are_both_verified(self):
        result = custody.audit(self.repo, self.authority)
        self.assertEqual(result["result"], "PASS")
        self.assertEqual([row["revision"] for row in result["specifications"]], ["1.1", "1.2"])
        self.assertEqual(result["protected_source"]["files"], 1)

    def test_altered_retained_manifest_is_rejected(self):
        self.retained.write_bytes(b'{"replacement": true}\n')
        with self.assertRaisesRegex(custody.CustodyError, "Retained revision"):
            custody.audit(self.repo, self.authority)

    def test_unrelated_commit_cannot_become_approved_ancestor(self):
        tree = self.git("rev-parse", "HEAD^{tree}").strip()
        other = self.git("commit-tree", tree, "-m", "Unrelated synthetic authority").strip()
        self.binding["approved_baseline"]["commit"] = other
        self.write_binding()
        with self.assertRaisesRegex(custody.CustodyError, "cannot be verified"):
            custody.audit(self.repo, self.authority)

    def test_source_inventory_and_authority_substitution_still_fails(self):
        self.source.write_bytes(b"Substituted source\n")
        self.inventory.write_text(json.dumps({"files_sha256": {"course/lesson.md": custody.sha256(self.source.read_bytes())}}))
        self.make_manifest("1.2")
        self.binding["active_specification"] = self.expected("1.2")
        self.binding["protected_source"]["inventory_sha256"] = custody.sha256(self.inventory.read_bytes())
        self.write_binding()
        with self.assertRaisesRegex(custody.CustodyError, "approved Git inventory"):
            custody.audit(self.repo, self.authority)



if __name__ == "__main__":
    unittest.main()
