"""Native Windows smoke probe for the owner-only S1 control boundary.

Run directly with the qualified Windows Python.  Stdout is one JSON document
with no credentials, paths, SIDs, or exception text.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from llm_foundations_companion import platform_security  # noqa: E402
from llm_foundations_companion.auth import AuthManager, EXPECTED_ORIGIN  # noqa: E402
from llm_foundations_companion.control import (  # noqa: E402
    ControlClient,
    ControlServer,
    _windows_apis,
)


CASE_IDS = (
    "S1-WIN-CONTROL-PAIR-STATUS",
    "S1-WIN-CONTROL-FILE-ACL",
    "S1-WIN-CONTROL-NO-TCP",
    "S1-WIN-CONTROL-SHUTDOWN",
    "S1-WIN-CONTROL-PARTIAL-PREPARE",
)


def _new_server(root: Path) -> tuple[ControlServer, str]:
    instance_id = str(uuid.uuid4())
    auth = AuthManager(instance_id)
    server = ControlServer(
        root,
        instance_id,
        os.getpid(),
        "c" * 43,
        {
            "status": lambda _: {"instance_id": instance_id, "pid": os.getpid()},
            "pair": lambda _: auth.issue_bootstrap(),
        },
    )
    return server, instance_id


def _pipe_absent(endpoint: str) -> bool:
    kernel32, _, _ = _windows_apis()
    return not bool(kernel32.WaitNamedPipeW(endpoint, 1))


def _summary(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ordered = [cases[case_id] for case_id in CASE_IDS]
    counts = {
        status: sum(case["status"] == status for case in ordered)
        for status in ("PASS", "FAIL", "NOT_RUN")
    }
    return {
        "schema": "llm-foundations-native-windows-control-probe-v1",
        "platform": "windows" if os.name == "nt" else os.name,
        "python_version": platform.python_version(),
        "cases": ordered,
        "counts": counts,
    }


def main() -> int:
    cases = {
        case_id: {"case_id": case_id, "status": "NOT_RUN"}
        for case_id in CASE_IDS
    }
    if os.name != "nt":
        cases[CASE_IDS[0]] = {
            "case_id": CASE_IDS[0],
            "status": "FAIL",
            "error_type": "WrongPlatform",
        }
        print(json.dumps(_summary(cases), sort_keys=True, separators=(",", ":")))
        return 1

    current = CASE_IDS[0]
    try:
        with tempfile.TemporaryDirectory(prefix="llmf-native-control-") as temporary:
            root = Path(temporary) / "normal"
            root.mkdir()
            server, instance_id = _new_server(root)
            descriptor = server.start()
            try:
                client = ControlClient(root)
                status = client.verify_instance()
                pair = client.pair()
                assert status == {"instance_id": instance_id, "pid": os.getpid()}
                assert pair["url"].startswith(f"{EXPECTED_ORIGIN}/#/connect/")
                cases[current]["status"] = "PASS"

                current = CASE_IDS[1]
                platform_security.check_private_file(
                    root / "runtime" / "control.json"
                )
                writers = platform_security._windows_writable_sids(
                    root / "runtime" / "control.json"
                )
                allowed = {
                    platform_security._current_windows_sid(),
                    "S-1-5-18",
                    "S-1-5-32-544",
                }
                assert writers <= allowed
                cases[current]["status"] = "PASS"

                current = CASE_IDS[2]
                assert descriptor.transport == "named_pipe"
                assert descriptor.endpoint.startswith(
                    r"\\.\pipe\llm-foundations-"
                )
                assert server._listener is None
                cases[current]["status"] = "PASS"
            finally:
                server.stop()

            current = CASE_IDS[3]
            assert not (root / "runtime" / "control.json").exists()
            assert _pipe_absent(descriptor.endpoint)
            cases[current]["status"] = "PASS"

            current = CASE_IDS[4]
            partial_root = Path(temporary) / "partial"
            blocked_control_path = partial_root / "runtime" / "control.json"
            blocked_control_path.mkdir(parents=True)
            partial, _ = _new_server(partial_root)
            partial_endpoint = partial.descriptor.endpoint
            prepare_failed = False
            try:
                partial.start()
            except (OSError, RuntimeError):
                prepare_failed = True
            finally:
                partial.stop()
            assert prepare_failed
            assert _pipe_absent(partial_endpoint)
            cases[current]["status"] = "PASS"
    except BaseException as exc:
        cases[current] = {
            "case_id": current,
            "status": "FAIL",
            "error_type": type(exc).__name__,
        }
        print(json.dumps(_summary(cases), sort_keys=True, separators=(",", ":")))
        return 1

    print(json.dumps(_summary(cases), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
