"""Failure controls for S0 report retention and complete denominators."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_s0_gate as gate


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_failed_initial_custody_retains_all_mandatory_gates(self):
        repo = self.root / "repo"
        repo.mkdir()
        output = self.root / "attempt-1"
        self.assertEqual(gate.main(["--repo-root", str(repo), "--report-dir", str(output), "--canonical-course", str(self.root / "canonical")]), 1)
        result = json.loads((output / "result.json").read_text())
        self.assertEqual(result["s0_gate"], "FAIL")
        self.assertEqual(result["product_qualification"], "NOT_RUN")
        self.assertEqual({row["gate_id"] for row in result["gates"]}, set(gate.GATE_PLAN))
        statuses = {row["gate_id"]: row["status"] for row in result["gates"]}
        self.assertEqual(statuses["python-runtime"], "PASS")
        self.assertEqual(statuses["custody-before"], "FAIL")
        self.assertTrue(all(status == "BLOCKED" for name, status in statuses.items()
                            if name not in {"python-runtime", "custody-before"}))

    def test_authoring_environment_rejects_package_drift(self):
        locks = self.root / "companion/locks"
        locks.mkdir(parents=True)
        (locks / "dev.requirements.txt").write_text("example==1.2.3\\n")
        with patch.object(gate.importlib.metadata, "version", return_value="1.2.4"):
            with self.assertRaisesRegex(ValueError, "expected 1.2.3, found 1.2.4"):
                gate.authoring_environment(self.root)
        with patch.object(gate.importlib.metadata, "version", return_value="1.2.3"):
            self.assertEqual(gate.authoring_environment(self.root)["packages"],
                             [{"name": "example", "version": "1.2.3"}])

    def test_existing_attempt_is_never_reused(self):
        repo = self.root / "repo"
        repo.mkdir()
        output = self.root / "attempt-1"
        output.mkdir()
        original = output / "result.json"
        original.write_bytes(b"retained original attempt\n")
        with self.assertRaises(FileExistsError):
            gate.main(["--repo-root", str(repo), "--report-dir", str(output), "--canonical-course", str(self.root / "canonical")])
        self.assertEqual(original.read_bytes(), b"retained original attempt\n")

    def test_failed_process_output_is_retained_with_digests(self):
        output = self.root / "attempt"
        output.mkdir()
        runner = gate.Runner(self.root, output)
        row = runner.command("negative-control", [
            sys.executable, "-c", "import sys; print('before failure'); print('error evidence', file=sys.stderr); sys.exit(7)"])
        self.assertEqual(row["status"], "FAIL")
        self.assertEqual(row["exit_code"], 7)
        for stream, expected in (("stdout", b"before failure"), ("stderr", b"error evidence")):
            raw = (output / row[stream]["path"]).read_bytes()
            self.assertIn(expected, raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row[stream]["sha256"])

    def test_unit_gate_rejects_zero_skipped_failed_and_error_denominators(self):
        report = self.root / "units.xml"
        for attributes in ('tests="0"', 'tests="5" skipped="1"', 'tests="5" failures="1"', 'tests="5" errors="1"'):
            report.write_text("<testsuites><testsuite " + attributes + "/></testsuites>")
            with self.subTest(attributes=attributes), self.assertRaises(ValueError):
                gate.unit_denominator(report)
        report.write_text('<testsuites><testsuite tests="5" failures="0" errors="0" skipped="0"/></testsuites>')
        self.assertEqual(gate.unit_denominator(report)["tests"], 5)


if __name__ == "__main__":
    unittest.main()
