"""Gate accounting preserves the real WSL skip rather than promoting it to PASS."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from run_s1_gate import unit_denominator


def junit(tmp_path, *, name="test_actual_windows_named_pipe_has_no_tcp_fallback"):
    path = tmp_path / "unit-tests.xml"
    path.write_text('<testsuites><testsuite><testcase classname="tests.test_api" name="test_health"/>'
        '<testcase classname="tests.test_auth_control" name="' + name + '">'
        '<skipped message="requires native Windows named pipes"/></testcase></testsuite></testsuites>')
    return path


def test_actual_pytest_classname_retains_native_skip_in_denominator(tmp_path):
    result = unit_denominator(junit(tmp_path), windows_passed=True)
    assert result["total"] == 2 and result["passed"] == 1 and result["not_run"] == 1
    assert result["cases"][1]["status"] == "NOT_RUN"


def test_native_probe_failure_cannot_excuse_the_skip(tmp_path):
    with pytest.raises(ValueError, match="Unexplained unit skips"):
        unit_denominator(junit(tmp_path), windows_passed=False)


def test_other_skipped_case_still_fails_even_with_native_probe_pass(tmp_path):
    with pytest.raises(ValueError, match="Unexplained unit skips"):
        unit_denominator(junit(tmp_path, name="unexpected_skipped_case"), windows_passed=True)
