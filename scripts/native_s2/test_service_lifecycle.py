"""Real installed-service lifecycle checks for the frozen S2 denominator."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import replace
import hashlib
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid


ORIGIN = "http://127.0.0.1:8765"
TERMINAL = frozenset({"completed", "failed", "interrupted"})
BYTE_TOKENIZER_SHA256 = (
    "0ab08b5f80c7c0853d6d2f1127f98837a8556c016dcc8b9552f0d868b4d6c3cf"
)
CHECKPOINT_FILES = frozenset(
    {
        "config.json",
        "manifest.json",
        "model.safetensors",
        "optimizer.safetensors",
        "rng.safetensors",
        "tokenizer.json",
        "trainer_state.json",
    }
)


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fixture_jsonl(total_text_bytes: int, *, prefix: str) -> bytes:
    """Build distinct document rows with an exact aggregate ASCII text size."""

    if total_text_bytes < 1:
        raise ValueError("fixture requires positive text bytes")
    rows = []
    remaining = total_text_bytes
    index = 0
    while remaining:
        size = min(40_000, remaining)
        pattern = hashlib.sha256(
            f"{prefix}:{index}".encode("utf-8")
        ).hexdigest()
        text = (pattern * ((size + len(pattern) - 1) // len(pattern)))[:size]
        rows.append(
            canonical(
                {
                    "record_id": f"{prefix}-{index:03d}",
                    "scenario_group_id": f"{prefix}-group-{index:03d}",
                    "text": text,
                }
            )
        )
        remaining -= size
        index += 1
    return b"\n".join(rows) + b"\n"


class InstalledService:
    """One real installed companion service and its authenticated API client."""

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root
        self.execution_parent = Path(
            tempfile.mkdtemp(prefix="llmf-s2-service-", dir="/tmp")
        ).resolve()
        self.root = self.execution_parent / "service-root"
        self.retained_root = self.artifact_root / "service-root"
        self.python = Path(sys.executable).absolute()
        self.environment = {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "PATH",
                "HOME",
                "USER",
                "LANG",
                "LC_ALL",
                "SYSTEMROOT",
                "WINDIR",
                "TEMP",
                "TMP",
            }
        }
        self.environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
        self.process: subprocess.Popen[bytes] | None = None
        self.session: dict[str, object] | None = None
        self.stdout_queue: queue.Queue[str] = queue.Queue()
        self.stdout_lines: list[str] = []
        self.stderr_blocks: list[bytes] = []
        self.readers: list[threading.Thread] = []

    def start(self) -> dict[str, object]:
        self.root.mkdir(parents=True, exist_ok=False)
        self.process = subprocess.Popen(
            [
                str(self.python),
                "-I",
                "-m",
                "llm_foundations_companion",
                "serve",
                "--storage",
                str(self.root),
                "--profile",
                "wsl-cpu",
            ],
            cwd=self.root.parent,
            env=self.environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self.process.stdout is not None and self.process.stderr is not None

        def stdout_reader() -> None:
            assert self.process is not None and self.process.stdout is not None
            for raw in self.process.stdout:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                self.stdout_lines.append(line)
                self.stdout_queue.put(line)

        def stderr_reader() -> None:
            assert self.process is not None and self.process.stderr is not None
            for block in iter(lambda: self.process.stderr.read(65_536), b""):
                self.stderr_blocks.append(block)

        self.readers = [
            threading.Thread(target=stdout_reader, daemon=True),
            threading.Thread(target=stderr_reader, daemon=True),
        ]
        for thread in self.readers:
            thread.start()
        deadline = time.monotonic() + 60.0
        bootstrap = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(
                    "Installed service exited during startup: "
                    + b"".join(self.stderr_blocks).decode("utf-8", "replace")[:4_000]
                )
            try:
                line = self.stdout_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if line.startswith(ORIGIN + "/#/connect/"):
                bootstrap = line.rsplit("/", 1)[1]
                break
        if bootstrap is None:
            raise AssertionError("Installed service did not publish a pairing grant")
        status, value, _ = self.request(
            "POST",
            "/api/v1/sessions",
            body={"bootstrap_secret": bootstrap},
            auth=False,
        )
        if status != 201 or not isinstance(value, dict):
            raise AssertionError(f"Session exchange failed: {status} {value!r}")
        self.session = value
        return value

    def stop(self) -> None:
        process = self.process
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
                raise AssertionError("Installed service did not stop within 45 seconds")
            for thread in self.readers:
                thread.join(timeout=3)
            self.process = None
        (self.artifact_root / "service.stdout.txt").write_text(
            "\n".join(self.stdout_lines) + ("\n" if self.stdout_lines else ""),
            encoding="utf-8",
        )
        (self.artifact_root / "service.stderr.txt").write_bytes(
            b"".join(self.stderr_blocks)
        )
        if self.root.is_dir():
            if self.retained_root.exists():
                raise AssertionError("Retained service root already exists")
            omitted_special_paths: list[str] = []

            def omit_special(directory: str, names: list[str]) -> list[str]:
                skipped = []
                base = Path(directory)
                for name in names:
                    mode = os.lstat(base / name).st_mode
                    if not stat.S_ISDIR(mode) and not stat.S_ISREG(mode):
                        skipped.append(name)
                        omitted_special_paths.append(
                            (base / name).relative_to(self.root).as_posix()
                        )
                return skipped

            shutil.copytree(
                self.root, self.retained_root, ignore=omit_special
            )
            (self.artifact_root / "service-execution-root.json").write_bytes(
                canonical(
                    {
                        "format": "llm-foundations-s2-service-root-v1",
                        "execution_root": str(self.root),
                        "retained_root": str(self.retained_root),
                        "omitted_special_paths": sorted(omitted_special_paths),
                    }
                )
                + b"\n"
            )
            temporary = Path(tempfile.gettempdir()).resolve()
            owned_parent = self.execution_parent.resolve(strict=True)
            owned_root = self.root.resolve(strict=True)
            if (
                temporary != Path("/tmp")
                or owned_parent.parent != temporary
                or not owned_parent.name.startswith("llmf-s2-service-")
                or owned_root != owned_parent / "service-root"
            ):
                raise AssertionError("Refusing to clean an unexpected service root")
            shutil.rmtree(owned_parent)

    def request(
        self,
        method: str,
        path: str,
        *,
        body: object | bytes | None = None,
        headers: dict[str, str] | None = None,
        auth: bool = True,
        idempotency_key: str | None = None,
        parse_json: bool = True,
    ) -> tuple[int, object, dict[str, str]]:
        values = {"Host": "127.0.0.1:8765", "Sec-Fetch-Site": "none"}
        if auth and self.session is not None:
            values["Authorization"] = "Bearer " + str(self.session["access_token"])
        if method in {"POST", "PUT", "PATCH", "DELETE"}:
            values["Origin"] = ORIGIN
            if auth and self.session is not None:
                values["X-LLMF-CSRF"] = str(self.session["csrf_token"])
            values["Idempotency-Key"] = idempotency_key or str(uuid.uuid4())
        raw_body = body
        if body is not None and not isinstance(body, bytes):
            raw_body = canonical(body)
            values["Content-Type"] = "application/json"
        values.update(headers or {})
        connection = http.client.HTTPConnection("127.0.0.1", 8765, timeout=30)
        try:
            connection.request(method, path, raw_body, values)
            response = connection.getresponse()
            raw = response.read()
            response_headers = {
                key.lower(): value for key, value in response.getheaders()
            }
        finally:
            connection.close()
        value: object = None
        if raw and parse_json:
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                value = raw
        elif raw:
            value = raw
        return response.status, value, response_headers

    def get(self, path: str) -> dict[str, object]:
        status, value, _ = self.request("GET", path)
        if status != 200 or not isinstance(value, dict):
            raise AssertionError(f"GET {path} failed: {status} {value!r}")
        return value

    def submit(
        self, request: dict[str, object], *, key: str | None = None
    ) -> dict[str, object]:
        status, value, _ = self.request(
            "POST", "/api/v1/jobs", body=request, idempotency_key=key
        )
        if status != 202 or not isinstance(value, dict):
            raise AssertionError(f"Job submission failed: {status} {value!r}")
        return value

    def wait_for(
        self,
        job_id: str,
        predicate: object,
        *,
        timeout: float = 300.0,
    ) -> tuple[dict[str, object], list[tuple[str, object]]]:
        deadline = time.monotonic() + timeout
        seen: list[tuple[str, object]] = []
        previous: tuple[str, object] | None = None
        while time.monotonic() < deadline:
            job = self.get("/api/v1/jobs/" + job_id)
            current = (str(job["state"]), job.get("phase"))
            if current != previous:
                seen.append(current)
                previous = current
            if predicate(job):
                return job, seen
            time.sleep(0.03)
        raise AssertionError(f"Timed out waiting for {job_id}; states={seen!r}")

    def run_job(
        self, request: dict[str, object], *, timeout: float = 300.0
    ) -> tuple[dict[str, object], list[tuple[str, object]]]:
        submitted = self.submit(request)
        job, seen = self.wait_for(
            str(submitted["job_id"]),
            lambda value: value["state"] in TERMINAL,
            timeout=timeout,
        )
        if job["state"] != "completed":
            raise AssertionError(f"Job did not complete: {job!r}")
        return job, seen

    def job_count(self) -> int:
        return len(self.get("/api/v1/jobs?limit=100")["items"])

    def job_events(self, job_id: str) -> list[dict[str, object]]:
        after = 0
        events: list[dict[str, object]] = []
        while True:
            page = self.get(
                f"/api/v1/jobs/{job_id}/events?after_cursor={after}&limit=100"
            )
            items = page["items"]
            if not isinstance(items, list):
                raise AssertionError("Job event page items are not a list")
            events.extend(dict(item) for item in items)
            next_after = page["next_after_cursor"]
            has_more = page["has_more"]
            if (
                not isinstance(next_after, int)
                or not isinstance(has_more, bool)
                or next_after < after
                or (has_more and next_after == after)
            ):
                raise AssertionError("Job event cursor did not advance monotonically")
            after = next_after
            if not has_more:
                return events

    def cancel(self, job_id: str) -> dict[str, object]:
        status, value, _ = self.request(
            "POST", f"/api/v1/jobs/{job_id}/cancel", body=None
        )
        if status != 200 or not isinstance(value, dict):
            raise AssertionError(f"Cancel failed: {status} {value!r}")
        return value

    @staticmethod
    def multipart(parts: list[tuple[str, str, bytes]]) -> tuple[bytes, dict[str, str]]:
        boundary = "llmf-s2-" + uuid.uuid4().hex
        output = bytearray()
        for name, media, raw in parts:
            output.extend(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{name}"; '
                    'filename="native-s2.bin"\r\n'
                    f"Content-Type: {media}\r\n\r\n"
                ).encode("ascii")
            )
            output.extend(raw)
            output.extend(b"\r\n")
        output.extend(f"--{boundary}--\r\n".encode("ascii"))
        return bytes(output), {
            "Content-Type": "multipart/form-data; boundary=" + boundary
        }

    def upload_dataset(
        self,
        *,
        name: str,
        train: bytes,
        validation: bytes,
    ) -> dict[str, object]:
        body, headers = self.multipart(
            [
                (
                    "metadata",
                    "application/json",
                    canonical({"name": name, "record_format": "document_text_v1"}),
                ),
                ("train", "application/x-ndjson", train),
                ("validation", "application/x-ndjson", validation),
            ]
        )
        status, value, _ = self.request(
            "POST", "/api/v1/datasets", body=body, headers=headers
        )
        if status != 201 or not isinstance(value, dict):
            raise AssertionError(f"Dataset upload failed: {status} {value!r}")
        return value

    def artifact_bytes(self, artifact_id: str) -> bytes:
        status, value, _ = self.request(
            "GET",
            f"/api/v1/artifacts/{artifact_id}/content",
            parse_json=False,
        )
        if status != 200 or not isinstance(value, bytes):
            raise AssertionError(f"Artifact read failed: {status} {value!r}")
        return value

    def checkpoint_rows(self, *, run_id: str | None = None) -> list[dict[str, object]]:
        items = self.get("/api/v1/checkpoints?limit=100")["items"]
        rows = [dict(value) for value in items]
        return rows if run_id is None else [
            value for value in rows if value.get("run_id") == run_id
        ]

    def checkpoint_files(self, checkpoint_id: str) -> dict[str, Path]:
        database = (self.root / "metadata.sqlite3").resolve()
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            rows = tuple(
                connection.execute(
                    """
                    SELECT file_name, artifact_id, sha256, size_bytes
                    FROM checkpoint_files
                    WHERE checkpoint_id = ?
                    ORDER BY file_name
                    """,
                    (checkpoint_id,),
                )
            )
        destination = self.root.parent / "checkpoints" / checkpoint_id
        destination.mkdir(parents=True, exist_ok=True)
        result: dict[str, Path] = {}
        for name, artifact_id, expected_digest, expected_size in rows:
            raw = self.artifact_bytes(str(artifact_id))
            if digest(raw) != expected_digest or len(raw) != expected_size:
                raise AssertionError(f"Checkpoint artifact identity differs: {name}")
            path = destination / str(name)
            if path.exists() and path.read_bytes() != raw:
                raise AssertionError(f"Retained checkpoint copy changed: {name}")
            if not path.exists():
                path.write_bytes(raw)
            result[str(name)] = path
        return result

    def register_fixture_artifact(
        self, raw: bytes, *, artifact_type: str
    ) -> dict[str, object]:
        from llm_foundations_companion.database import Database
        from llm_foundations_companion.registry import Registry

        database = Database(self.root)
        database.initialize()
        if database.read_only:
            raise AssertionError("Fixture registry handle entered recovery")
        assert self.session is not None
        registry = Registry(
            database,
            self.root,
            str(self.session["instance_id"]),
            b"S2-validation-cursor-key-00000000",
        )
        owner = str(uuid.uuid4())
        registry.reserve_capacity(
            owner_kind="upload",
            owner_id=owner,
            byte_count=len(raw) + 65_536,
            artifact_rows=1,
            reservation_id=owner,
        )
        try:
            return registry.register_stream(
                raw,
                owner,
                artifact_type=artifact_type,
                display_name="S2 accepted-worker stale fixture",
                media_type="application/json",
                preview_policy="metadata_only",
                origin="locally_created",
                expected_sha256=digest(raw),
                reservation_id=owner,
            )
        finally:
            registry.release_capacity_now(owner)

    def cli_request(
        self, request_path: Path, *, key: str
    ) -> tuple[dict[str, object], list[str]]:
        argv = [
            str(self.python),
            "-I",
            "-m",
            "llm_foundations_companion",
            "request",
            "--storage",
            str(self.root),
            "--file",
            str(request_path),
            "--idempotency-key",
            key,
        ]
        result = subprocess.run(
            argv,
            cwd=self.root.parent,
            env=self.environment,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise AssertionError(
                "CLI request failed: "
                + (result.stderr or result.stdout).decode("utf-8", "replace")[:4_000]
            )
        return json.loads(result.stdout), argv


class ServiceLifecycle(unittest.TestCase):
    service: InstalledService
    dataset: dict[str, object]
    byte_tokenizer: dict[str, object]
    full_job: dict[str, object]
    full_checkpoint_id: str
    resumed_checkpoint_id: str
    preview_request: dict[str, object]
    preview_result: dict[str, object]
    held_checkpoint_id: str

    @classmethod
    def setUpClass(cls) -> None:
        artifact_root = Path(os.environ["LLMF_S2_NATIVE_ARTIFACT_ROOT"]).resolve()
        cls.service = InstalledService(artifact_root)
        try:
            cls.service.start()
            fixture_root = (
                Path(__file__).resolve().parents[2]
                / "docs/specs/intermediate-v1/fixtures/data/materialized/data-clinic-v1"
            )
            cls.dataset = cls.service.upload_dataset(
                name="S2 data clinic",
                train=(fixture_root / "train.jsonl").read_bytes(),
                validation=(fixture_root / "validation.jsonl").read_bytes(),
            )
        except BaseException:
            cls.service.stop()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        cls.service.stop()

    @classmethod
    def tiny_request(
        cls,
        *,
        dataset_id: str,
        tokenizer_id: str,
        steps: int,
        eval_every: int,
    ) -> dict[str, object]:
        return {
            "operation": "tiny_train",
            "dataset_id": dataset_id,
            "tokenizer_id": tokenizer_id,
            "steps": steps,
            "eval_every": eval_every,
            "batch_size": 2,
            "learning_rate": 0.001,
            "seed": 17,
            "context": 8,
            "width": 16,
            "heads": 1,
            "layers": 1,
            "architecture_profile_id": "tiny-v2-standard-v1",
        }

    def require_full_training(self) -> None:
        if not hasattr(self.__class__, "full_checkpoint_id"):
            self.skipTest(
                "NOT_RUN: S2-NATIVE-006 did not establish the shared real-training "
                "checkpoint required by this case."
            )

    def require_held_checkpoint(self) -> None:
        if not hasattr(self.__class__, "held_checkpoint_id"):
            self.skipTest(
                "NOT_RUN: S2-NATIVE-020 did not establish the held step-25 "
                "checkpoint required by this case."
            )

    @staticmethod
    def result(job: dict[str, object]) -> dict[str, object]:
        value = job.get("result")
        if not isinstance(value, dict):
            raise AssertionError(f"Completed job lacks a result: {job!r}")
        return value

    @staticmethod
    def metric_rows(raw: bytes) -> list[dict[str, object]]:
        return [json.loads(line) for line in raw.splitlines() if line]

    def test_s2_native_003_standalone_byte_tokenizer(self) -> None:
        job, states = self.service.run_job(
            {
                "operation": "tokenizer_train",
                "dataset_id": self.dataset["dataset_id"],
                "vocab_size": 257,
                "seed": 17,
                "tokenizer_profile_id": "byte-v1",
            }
        )
        result = self.result(job)
        raw = self.service.artifact_bytes(str(result["artifact_ids"][0]))
        tokenizer = json.loads(raw)
        self.assertEqual(
            {
                "operation": result["operation"],
                "tokenizer_type": result["tokenizer_type"],
                "vocab_size": result["vocab_size"],
                "tokenizer_sha256": result["tokenizer_sha256"],
            },
            {
                "operation": "tokenizer_train",
                "tokenizer_type": "byte",
                "vocab_size": 257,
                "tokenizer_sha256": BYTE_TOKENIZER_SHA256,
            },
        )
        self.assertEqual(tokenizer["format"], "teaching-byte-bpe-v1")
        self.assertEqual(tokenizer["merges"], [])
        self.assertEqual(
            digest(json.dumps(tokenizer, sort_keys=True).encode("utf-8")),
            BYTE_TOKENIZER_SHA256,
        )
        self.__class__.byte_tokenizer = result
        self.s2_observation = {
            "job_id": job["job_id"],
            "states": states,
            "tokenizer_id": result["tokenizer_id"],
            "tokenizer_sha256": result["tokenizer_sha256"],
            "merges": 0,
            "artifact_sha256": digest(raw),
        }

    def test_s2_native_004_deterministic_bpe_tokenizer(self) -> None:
        request = {
            "operation": "tokenizer_train",
            "dataset_id": self.dataset["dataset_id"],
            "vocab_size": 280,
            "seed": 17,
            "tokenizer_profile_id": "byte-bpe-v1",
        }
        first_job, _ = self.service.run_job(request)
        second_job, _ = self.service.run_job(request)
        first = self.result(first_job)
        second = self.result(second_job)
        first_raw = self.service.artifact_bytes(str(first["artifact_ids"][0]))
        second_raw = self.service.artifact_bytes(str(second["artifact_ids"][0]))
        first_value = json.loads(first_raw)
        second_value = json.loads(second_raw)
        self.assertEqual(first_raw, second_raw)
        self.assertEqual(first_value, second_value)
        self.assertEqual(first["tokenizer_sha256"], second["tokenizer_sha256"])
        self.assertEqual(first["vocab_size"], second["vocab_size"])
        self.assertEqual(first["tokenizer_type"], second["tokenizer_type"])
        self.assertEqual(first["tokenizer_type"], "byte_bpe")
        self.assertEqual(len(first_value["merges"]), int(first["vocab_size"]) - 257)
        self.assertEqual(
            first_value["merges"],
            second_value["merges"],
        )
        for index, pair in enumerate(first_value["merges"]):
            self.assertEqual(len(pair), 2)
            self.assertTrue(all(0 <= token_id < 257 + index for token_id in pair))
            self.assertNotIn(256, pair)
        self.s2_observation = {
            "job_ids": [first_job["job_id"], second_job["job_id"]],
            "tokenizer_ids": [first["tokenizer_id"], second["tokenizer_id"]],
            "vocab_size": first["vocab_size"],
            "merge_count": len(first_value["merges"]),
            "tokenizer_sha256": first["tokenizer_sha256"],
            "artifact_sha256": digest(first_raw),
        }

    def test_s2_native_005_training_text_byte_cap(self) -> None:
        over = self.service.upload_dataset(
            name="S2 over text cap",
            train=fixture_jsonl(200_001, prefix="over-train"),
            validation=fixture_jsonl(11, prefix="over-validation"),
        )
        before = self.service.job_count()
        rejected = []
        requests = [
            {
                "operation": "tokenizer_train",
                "dataset_id": over["dataset_id"],
                "vocab_size": 257,
                "seed": 17,
                "tokenizer_profile_id": "byte-v1",
            },
            self.tiny_request(
                dataset_id=str(over["dataset_id"]),
                tokenizer_id=str(self.byte_tokenizer["tokenizer_id"]),
                steps=1,
                eval_every=1,
            ),
        ]
        for request in requests:
            status, value, _ = self.service.request(
                "POST", "/api/v1/jobs", body=request
            )
            self.assertEqual(status, 400, value)
            self.assertIsInstance(value, dict)
            error = value["error"]
            self.assertEqual(error["reason_code"], "PAYLOAD_TOO_LARGE")
            rejected.append(
                {
                    "operation": request["operation"],
                    "status": status,
                    "reason_code": error["reason_code"],
                }
            )
        self.assertEqual(self.service.job_count(), before)

        at_cap = self.service.upload_dataset(
            name="S2 exact text cap plus validation",
            train=fixture_jsonl(200_000, prefix="cap-train"),
            validation=fixture_jsonl(50_000, prefix="cap-validation"),
        )
        tokenizer_job, _ = self.service.run_job(
            {
                "operation": "tokenizer_train",
                "dataset_id": at_cap["dataset_id"],
                "vocab_size": 257,
                "seed": 17,
                "tokenizer_profile_id": "byte-v1",
            }
        )
        at_cap_tokenizer = self.result(tokenizer_job)
        accepted_train, _ = self.service.run_job(
            self.tiny_request(
                dataset_id=str(at_cap["dataset_id"]),
                tokenizer_id=str(at_cap_tokenizer["tokenizer_id"]),
                steps=1,
                eval_every=1,
            ),
            timeout=300,
        )
        self.assertEqual(accepted_train["state"], "completed")
        self.s2_observation = {
            "train_limit_bytes": 200_000,
            "validation_bytes_excluded": 50_000,
            "rejections": rejected,
            "job_count_unchanged_after_rejections": before,
            "at_cap_tokenizer_job": tokenizer_job["job_id"],
            "at_cap_training_job": accepted_train["job_id"],
        }

    def test_s2_native_006_real_fifty_update_training(self) -> None:
        request = self.tiny_request(
            dataset_id=str(self.dataset["dataset_id"]),
            tokenizer_id=str(self.byte_tokenizer["tokenizer_id"]),
            steps=50,
            eval_every=25,
        )
        job, states = self.service.run_job(request, timeout=600)
        result = self.result(job)
        self.assertEqual(result["operation"], "tiny_train")
        self.assertEqual(result["completed_step"], 50)
        self.assertEqual(result["requested_final_step"], 50)
        metrics_raw = self.service.artifact_bytes(
            str(result["metrics_artifact_id"])
        )
        metrics = self.metric_rows(metrics_raw)
        self.assertEqual(len(metrics), 156)
        self.assertEqual(
            [int(row["sequence"]) for row in metrics], list(range(156))
        )
        numeric_values = []
        for row in metrics:
            value = row["value"]
            self.assertIsInstance(value, (int, float))
            self.assertNotIsInstance(value, bool)
            self.assertTrue(math.isfinite(float(value)), (row["name"], value))
            numeric_values.append(float(value))
        checkpoints = sorted(
            self.service.checkpoint_rows(run_id=str(result["run_id"])),
            key=lambda value: value["step"],
        )
        self.assertEqual([value["step"] for value in checkpoints], [0, 25, 50])
        self.assertEqual(
            str(result["last_checkpoint_id"]), str(checkpoints[-1]["checkpoint_id"])
        )
        self.__class__.full_job = job
        self.__class__.full_checkpoint_id = str(result["last_checkpoint_id"])
        self.s2_observation = {
            "job_id": job["job_id"],
            "run_id": result["run_id"],
            "model_id": result["model_id"],
            "completed_step": result["completed_step"],
            "checkpoint_steps": [value["step"] for value in checkpoints],
            "metric_rows": len(metrics),
            "finite_numeric_values": len(numeric_values),
            "states": states,
        }

    def test_s2_native_007_safe_checkpoint_format_and_identity(self) -> None:
        self.require_full_training()
        from llm_foundations_companion.tiny_v2.checkpoint import (
            CheckpointExpectations,
            validate_checkpoint,
        )

        records = {
            str(value["checkpoint_id"]): value
            for value in self.service.checkpoint_rows()
        }
        record = records[self.full_checkpoint_id]
        files = self.service.checkpoint_files(self.full_checkpoint_id)
        self.assertEqual(set(files), CHECKPOINT_FILES)
        self.assertTrue(all(path.is_file() and not path.is_symlink() for path in files.values()))
        self.assertFalse(
            any(path.suffix in {".pt", ".pth", ".pkl", ".pickle", ".bin"} for path in files.values())
        )
        manifest_raw = files["manifest.json"].read_bytes()
        manifest = json.loads(manifest_raw)
        self.assertEqual(
            set(manifest), {"format", "portability", "files", "aliases"}
        )
        self.assertEqual(manifest["format"], "tiny-v2-checkpoint-v1")
        self.assertEqual(manifest["portability"], "resume")
        self.assertEqual(manifest["aliases"], {})
        self.assertEqual(
            {item["name"] for item in manifest["files"]},
            CHECKPOINT_FILES - {"manifest.json"},
        )
        for item in manifest["files"]:
            raw = files[item["name"]].read_bytes()
            self.assertEqual((item["size_bytes"], item["sha256"]), (len(raw), digest(raw)))
        validated = validate_checkpoint(
            files,
            expected=CheckpointExpectations(
                manifest_sha256=str(record["sha256"]),
                completed_global_step=50,
                runtime_profile="wsl-cpu",
                device="cpu",
            ),
            for_resume=True,
        )
        self.assertEqual(validated.manifest_sha256, digest(manifest_raw))
        self.assertIsNotNone(validated.optimizer_tensors)
        self.assertIsNotNone(validated.rng_tensors)
        self.s2_observation = {
            "checkpoint_id": self.full_checkpoint_id,
            "manifest_sha256": validated.manifest_sha256,
            "files": [
                {
                    "name": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": digest(path.read_bytes()),
                }
                for name, path in sorted(files.items())
            ],
            "aliases": manifest["aliases"],
        }

    def test_s2_native_008_uninterrupted_resume_parity(self) -> None:
        self.require_full_training()
        from llm_foundations_companion.tiny_v2.checkpoint import load_checkpoint
        from llm_foundations_companion.tiny_v2.decoding import generate_tokens

        first_job, _ = self.service.run_job(
            self.tiny_request(
                dataset_id=str(self.dataset["dataset_id"]),
                tokenizer_id=str(self.byte_tokenizer["tokenizer_id"]),
                steps=25,
                eval_every=25,
            ),
            timeout=600,
        )
        first = self.result(first_job)
        resumed_job, _ = self.service.run_job(
            {
                "operation": "tiny_resume",
                "checkpoint_id": first["last_checkpoint_id"],
                "additional_steps": 25,
            },
            timeout=600,
        )
        resumed = self.result(resumed_job)
        self.assertEqual(resumed["completed_step"], 50)
        self.assertEqual(resumed["parent_checkpoint_id"], first["last_checkpoint_id"])
        uninterrupted_files = self.service.checkpoint_files(self.full_checkpoint_id)
        resumed_files = self.service.checkpoint_files(
            str(resumed["last_checkpoint_id"])
        )
        tensor_names = (
            "model.safetensors",
            "optimizer.safetensors",
            "rng.safetensors",
        )
        for name in tensor_names:
            self.assertEqual(
                uninterrupted_files[name].read_bytes(),
                resumed_files[name].read_bytes(),
                name,
            )
        uninterrupted_metrics = self.metric_rows(
            self.service.artifact_bytes(
                str(self.result(self.full_job)["metrics_artifact_id"])
            )
        )
        resumed_metrics = self.metric_rows(
            self.service.artifact_bytes(str(resumed["metrics_artifact_id"]))
        )
        def step_50_evaluation_values(
            rows: list[dict[str, object]],
        ) -> dict[str, object]:
            values = {}
            for name in ("train_nll_token", "validation_nll_token"):
                matches = [
                    row
                    for row in rows
                    if row["step"] == 50
                    and row["name"] == name
                    and row["unit"] == "nats_per_token"
                    and row["protocol_id"]
                    == "tiny-v2-fixed-eight-batches-v1"
                ]
                self.assertEqual(len(matches), 1, (name, matches))
                values[name] = matches[0]["value"]
            return values

        uninterrupted_50 = step_50_evaluation_values(uninterrupted_metrics)
        resumed_50 = step_50_evaluation_values(resumed_metrics)
        self.assertEqual(uninterrupted_50, resumed_50)
        uninterrupted = load_checkpoint(
            uninterrupted_files, device="cpu", for_resume=True
        )
        continued = load_checkpoint(resumed_files, device="cpu", for_resume=True)
        arguments = {
            "prompt": "Mira checked the result.",
            "context": uninterrupted.config.context,
            "max_new_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 17,
            "device": "cpu",
        }
        first_generation = generate_tokens(
            uninterrupted.model, uninterrupted.tokenizer, **arguments
        )
        second_generation = generate_tokens(
            continued.model, continued.tokenizer, **arguments
        )
        self.assertEqual(
            first_generation.generated_token_ids,
            second_generation.generated_token_ids,
        )
        self.__class__.resumed_checkpoint_id = str(resumed["last_checkpoint_id"])
        self.s2_observation = {
            "uninterrupted_job_id": self.full_job["job_id"],
            "split_job_id": first_job["job_id"],
            "resume_job_id": resumed_job["job_id"],
            "parent_checkpoint_id": first["last_checkpoint_id"],
            "uninterrupted_checkpoint_id": self.full_checkpoint_id,
            "resumed_checkpoint_id": resumed["last_checkpoint_id"],
            "identical_tensor_files": list(tensor_names),
            "step_50_validation_nll_token": resumed_50[
                "validation_nll_token"
            ],
            "greedy_token_ids": list(first_generation.generated_token_ids),
        }

    def test_s2_native_010_evaluation_preserves_weights(self) -> None:
        self.require_full_training()
        before_files = self.service.checkpoint_files(self.full_checkpoint_id)
        before = {
            name: digest(path.read_bytes()) for name, path in before_files.items()
        }
        before_checkpoints = len(self.service.checkpoint_rows())
        request = {
            "operation": "evaluate",
            "subjects": [
                {
                    "kind": "tiny_checkpoint",
                    "checkpoint_id": self.full_checkpoint_id,
                }
            ],
            "dataset_id": self.dataset["dataset_id"],
            "split": "validation",
            "evaluation_profile_id": "tiny-nll-per-byte-v1",
        }
        job, states = self.service.run_job(request, timeout=600)
        result = self.result(job)
        self.assertEqual(result["operation"], "evaluate")
        self.assertEqual(result["subjects"], request["subjects"])
        self.assertIsNone(result["paired_artifact_id"])
        metric_rows = self.metric_rows(
            self.service.artifact_bytes(str(result["metrics_artifact_id"]))
        )
        record_rows = self.metric_rows(
            self.service.artifact_bytes(str(result["records_artifact_id"]))
        )
        self.assertTrue(metric_rows and record_rows)
        self.assertTrue(
            all(
                math.isfinite(float(row["value"]))
                and row["unit"] == "nats_per_utf8_byte"
                for row in metric_rows
            )
        )
        after_files = self.service.checkpoint_files(self.full_checkpoint_id)
        after = {
            name: digest(path.read_bytes()) for name, path in after_files.items()
        }
        self.assertEqual(after, before)
        self.assertEqual(len(self.service.checkpoint_rows()), before_checkpoints)
        self.s2_observation = {
            "job_id": job["job_id"],
            "evaluation_id": result["evaluation_id"],
            "states": states,
            "metric_rows": len(metric_rows),
            "record_rows": len(record_rows),
            "checkpoint_count_unchanged": before_checkpoints,
            "checkpoint_file_sha256": before,
        }

    def test_s2_native_011_preview_tokenizer_only(self) -> None:
        before = {
            name: digest(path.read_bytes())
            for name, path in self.service.checkpoint_files(
                self.full_checkpoint_id
            ).items()
        }
        before_checkpoints = len(self.service.checkpoint_rows())
        request = {
            "operation": "context_preview",
            "backend": "tiny",
            "checkpoint_id": self.full_checkpoint_id,
            "prompt": "Hello\nworld",
            "max_new_tokens": 4,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 17,
        }
        job, states = self.service.run_job(request)
        result = self.result(job)
        preview_raw = self.service.artifact_bytes(
            str(result["preview_artifact_id"])
        )
        preview = json.loads(preview_raw)
        expected = {
            key: value
            for key, value in result.items()
            if key
            not in {
                "operation",
                "preview_artifact_id",
                "context_preview_digest",
                "artifact_ids",
            }
        }
        self.assertEqual(preview, expected)
        self.assertEqual(digest(preview_raw), result["context_preview_digest"])
        self.assertEqual(result["backend"], "tiny")
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["tokenizer_sha256"], BYTE_TOKENIZER_SHA256)
        expected_preview = {
            "original_input_token_count": 11,
            "input_token_ids": [108, 111, 10, 119, 111, 114, 108, 100],
            "input_token_count": 8,
            "serialized_text": "lo\nworld",
            "cropped_input_tokens": 3,
            "effective_context_budget": 8,
        }
        self.assertEqual(
            {key: result[key] for key in expected_preview}, expected_preview
        )
        after = {
            name: digest(path.read_bytes())
            for name, path in self.service.checkpoint_files(
                self.full_checkpoint_id
            ).items()
        }
        self.assertEqual(after, before)
        self.assertEqual(len(self.service.checkpoint_rows()), before_checkpoints)
        self.__class__.preview_request = request
        self.__class__.preview_result = result
        self.s2_observation = {
            "job_id": job["job_id"],
            "states": states,
            "preview_artifact_id": result["preview_artifact_id"],
            "context_preview_digest": result["context_preview_digest"],
            "original_input_token_count": result[
                "original_input_token_count"
            ],
            "input_token_ids": result["input_token_ids"],
            "cropped_input_tokens": result["cropped_input_tokens"],
            "serialized_text": result["serialized_text"],
            "checkpoint_files_unchanged": before,
            "checkpoint_count_unchanged": before_checkpoints,
        }

    def test_s2_native_012_matching_preview_generation(self) -> None:
        from llm_foundations_companion.tiny_v2.checkpoint import load_checkpoint
        from llm_foundations_companion.tiny_v2.decoding import generate_tokens

        files = self.service.checkpoint_files(self.full_checkpoint_id)
        before = {name: digest(path.read_bytes()) for name, path in files.items()}
        request = {
            key: value
            for key, value in self.preview_request.items()
            if key != "backend"
        }
        request["operation"] = "generate"
        request["preview_artifact_id"] = self.preview_result["preview_artifact_id"]
        request["context_preview_digest"] = self.preview_result[
            "context_preview_digest"
        ]
        job, states = self.service.run_job(request, timeout=300)
        result = self.result(job)
        artifact = json.loads(
            self.service.artifact_bytes(str(result["artifact_ids"][0]))
        )
        loaded = load_checkpoint(files, device="cpu")
        direct = generate_tokens(
            loaded.model,
            loaded.tokenizer,
            str(request["prompt"]),
            context=loaded.config.context,
            max_new_tokens=int(request["max_new_tokens"]),
            temperature=float(request["temperature"]),
            top_p=float(request["top_p"]),
            seed=int(request["seed"]),
            device="cpu",
        )
        self.assertEqual(artifact["output_token_ids"], list(direct.generated_token_ids))
        self.assertEqual(result["generated_text"], direct.generated_text)
        self.assertEqual(result["generated_token_count"], direct.generated_token_count)
        self.assertEqual(result["stop_reason"], direct.stop_reason)
        self.assertEqual(result["serialized_input"], "lo\nworld")
        after = {
            name: digest(path.read_bytes())
            for name, path in self.service.checkpoint_files(
                self.full_checkpoint_id
            ).items()
        }
        self.assertEqual(after, before)
        self.s2_observation = {
            "job_id": job["job_id"],
            "states": states,
            "generation_artifact_id": result["artifact_ids"][0],
            "output_token_ids": artifact["output_token_ids"],
            "generated_token_count": result["generated_token_count"],
            "stop_reason": result["stop_reason"],
            "checkpoint_files_unchanged": before,
        }

    def test_s2_native_013_canonical_numeric_preview_spelling(self) -> None:
        integer_request = dict(self.preview_request)
        integer_request["top_p"] = 1
        float_request = dict(self.preview_request)
        float_request["top_p"] = 1.0
        integer_job, _ = self.service.run_job(integer_request)
        float_job, _ = self.service.run_job(float_request)
        integer_result = self.result(integer_job)
        float_result = self.result(float_job)
        self.assertEqual(
            integer_result["canonical_generation_request_sha256"],
            float_result["canonical_generation_request_sha256"],
        )
        self.assertEqual(
            integer_result["context_preview_digest"],
            float_result["context_preview_digest"],
        )
        generation = {
            key: value for key, value in float_request.items() if key != "backend"
        }
        generation.update(
            operation="generate",
            preview_artifact_id=integer_result["preview_artifact_id"],
            context_preview_digest=integer_result["context_preview_digest"],
        )
        generated_job, _ = self.service.run_job(generation)
        self.assertEqual(generated_job["state"], "completed")
        self.s2_observation = {
            "integer_preview_job": integer_job["job_id"],
            "float_preview_job": float_job["job_id"],
            "generation_job": generated_job["job_id"],
            "canonical_generation_request_sha256": integer_result[
                "canonical_generation_request_sha256"
            ],
            "context_preview_digest": integer_result["context_preview_digest"],
        }

    def test_s2_native_014_stale_preview_prequeue_matrix(self) -> None:
        base = {
            key: value
            for key, value in self.preview_request.items()
            if key != "backend"
        }
        base.update(
            operation="generate",
            preview_artifact_id=self.preview_result["preview_artifact_id"],
            context_preview_digest=self.preview_result["context_preview_digest"],
        )
        variants: dict[str, dict[str, object]] = {}
        for label, updates in (
            ("prompt", {"prompt": "changed"}),
            ("seed", {"seed": 18}),
            ("budget", {"max_new_tokens": 5}),
            ("subject", {"checkpoint_id": self.resumed_checkpoint_id}),
            ("digest", {"context_preview_digest": "0" * 64}),
            ("preview", {"preview_artifact_id": str(uuid.uuid4())}),
        ):
            value = dict(base)
            value.update(updates)
            variants[label] = value
        before = self.service.job_count()
        results = {}
        for label, request in variants.items():
            status, value, _ = self.service.request(
                "POST", "/api/v1/jobs", body=request
            )
            self.assertEqual(status, 400, (label, value))
            self.assertIsInstance(value, dict)
            error = value["error"]
            self.assertEqual(error["code"], "VALIDATION_FAILED")
            self.assertEqual(error["reason_code"], "CONTEXT_PREVIEW_STALE")
            results[label] = {
                "status": status,
                "code": error["code"],
                "reason_code": error["reason_code"],
            }
        self.assertEqual(self.service.job_count(), before)
        self.s2_observation = {
            "matrix": results,
            "job_count_before": before,
            "job_count_after": self.service.job_count(),
        }

    def test_s2_native_015_accepted_worker_preview_revalidation(self) -> None:
        preview = json.loads(
            self.service.artifact_bytes(
                str(self.preview_result["preview_artifact_id"])
            )
        )
        changed = dict(preview)
        changed_ids = list(changed["input_token_ids"])
        changed_ids[0] = (int(changed_ids[0]) + 1) % 256
        changed["input_token_ids"] = changed_ids
        changed_raw = canonical(changed)
        forged = self.service.register_fixture_artifact(
            changed_raw, artifact_type="context_preview"
        )
        before = {
            name: digest(path.read_bytes())
            for name, path in self.service.checkpoint_files(
                self.full_checkpoint_id
            ).items()
        }
        request = {
            key: value
            for key, value in self.preview_request.items()
            if key != "backend"
        }
        request.update(
            operation="generate",
            preview_artifact_id=forged["artifact_id"],
            context_preview_digest=digest(changed_raw),
        )
        submitted = self.service.submit(request)
        failed, states = self.service.wait_for(
            str(submitted["job_id"]),
            lambda value: value["state"] in TERMINAL,
        )
        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["error"]["code"], "CONTEXT_PREVIEW_STALE")
        self.assertIsNone(failed["result"])
        self.assertEqual(failed["step"], 0)
        self.assertIsNone(failed["checkpoint_boundary"])
        self.assertEqual(
            failed["log_artifact_ids"], {"stdout": None, "stderr": None}
        )
        job_artifacts = self.service.get(
            f"/api/v1/artifacts?job_id={failed['job_id']}&limit=100"
        )["items"]
        job_artifacts.sort(key=lambda artifact: (artifact["type"], artifact["artifact_id"]))
        self.assertEqual(failed["committed_artifact_count"], len(job_artifacts))
        self.assertEqual(
            {
                (artifact["type"], artifact["display_name"])
                for artifact in job_artifacts
            },
            {
                ("job_request", "job-request.json"),
                ("run_result", "run-result.json"),
            },
        )
        self.assertTrue(
            all(artifact["job_id"] == failed["job_id"] for artifact in job_artifacts)
        )
        artifacts_by_type = {
            artifact["type"]: artifact for artifact in job_artifacts
        }
        self.assertEqual(len(artifacts_by_type), len(job_artifacts))
        request_artifact = artifacts_by_type["job_request"]
        run_result_artifact = artifacts_by_type["run_result"]
        self.assertEqual(
            request_artifact["artifact_id"], failed["request"]["artifact_id"]
        )
        request_raw = self.service.artifact_bytes(request_artifact["artifact_id"])
        self.assertEqual(request_raw, canonical(request))
        self.assertEqual(request_artifact["sha256"], digest(request_raw))
        self.assertEqual(
            failed["request"]["canonical_sha256"], digest(request_raw)
        )
        run_result_raw = self.service.artifact_bytes(
            run_result_artifact["artifact_id"]
        )
        self.assertEqual(
            run_result_raw,
            canonical(
                {
                    "error": "CONTEXT_PREVIEW_STALE",
                    "finished_at": failed["finished_at"],
                    "last_durable_checkpoint_step": None,
                    "last_observed_step": 0,
                    "requested_final_step": 0,
                    "run_id": failed["run_id"],
                    "status": "failed",
                }
            ),
        )
        self.assertEqual(run_result_artifact["sha256"], digest(run_result_raw))
        after = {
            name: digest(path.read_bytes())
            for name, path in self.service.checkpoint_files(
                self.full_checkpoint_id
            ).items()
        }
        self.assertEqual(after, before)
        self.s2_observation = {
            "job_id": failed["job_id"],
            "states": states,
            "forged_preview_artifact_id": forged["artifact_id"],
            "forged_preview_sha256": digest(changed_raw),
            "error_code": failed["error"]["code"],
            "committed_artifact_count": failed["committed_artifact_count"],
            "committed_artifact_types": sorted(
                artifact["type"] for artifact in job_artifacts
            ),
            "checkpoint_files_unchanged": before,
        }

    def test_s2_native_017_incompatible_checkpoint_identity_matrix(self) -> None:
        self.require_full_training()
        from llm_foundations_companion.tiny_v2.checkpoint import (
            CheckpointExpectations,
            CheckpointIncompatible,
            validate_checkpoint,
        )

        files = self.service.checkpoint_files(self.full_checkpoint_id)
        validated = validate_checkpoint(files, for_resume=True)
        assert validated.trainer_state is not None
        bindings = validated.trainer_state.dataset_bindings
        wrong_bindings = (
            replace(bindings[0], sha256="0" * 64),
            bindings[1],
        )
        matrix = {
            "dataset": CheckpointExpectations(dataset_bindings=wrong_bindings),
            "tokenizer": CheckpointExpectations(tokenizer_sha256="0" * 64),
            "config": CheckpointExpectations(config_sha256="0" * 64),
            "lock": CheckpointExpectations(dependency_lock_sha256="0" * 64),
            "profile": CheckpointExpectations(runtime_profile="win-cpu"),
        }
        before = self.service.job_count()
        messages = {}
        for label, expected in matrix.items():
            with self.assertRaises(CheckpointIncompatible) as caught:
                validate_checkpoint(files, expected=expected, for_resume=True)
            messages[label] = str(caught.exception)
        self.assertEqual(self.service.job_count(), before)
        self.s2_observation = {
            "checkpoint_id": self.full_checkpoint_id,
            "rejections": messages,
            "job_count_unchanged": before,
        }

    def test_s2_native_020_cancel_prescribed_step_twenty_five_hold(self) -> None:
        self.require_full_training()
        request = self.tiny_request(
            dataset_id=str(self.dataset["dataset_id"]),
            tokenizer_id=str(self.byte_tokenizer["tokenizer_id"]),
            steps=50,
            eval_every=25,
        )
        request.update(
            exercise_profile_id="tiny-v2-diagnosis-v1",
            curriculum_hold_after_step=25,
        )
        submitted = self.service.submit(request)
        held, before_cancel = self.service.wait_for(
            str(submitted["job_id"]),
            lambda value: (
                value.get("phase") == "cancellable_hold"
                or value["state"] in TERMINAL
            ),
            timeout=600,
        )
        self.assertEqual(held.get("phase"), "cancellable_hold", held)
        boundary = held["checkpoint_boundary"]
        self.assertIsInstance(boundary, dict)
        self.assertEqual(boundary["step"], 25)
        self.assertEqual(held["step"], 25)
        self.assertIsNotNone(held["hold_deadline_at"])
        cancelling = self.service.cancel(str(held["job_id"]))
        self.assertIn(cancelling["state"], {"cancelling", "interrupted"})
        terminal, after_cancel = self.service.wait_for(
            str(held["job_id"]),
            lambda value: value["state"] in TERMINAL,
            timeout=120,
        )
        self.assertEqual(terminal["state"], "interrupted")
        self.assertEqual(terminal["terminal_reason"], "user_cancelled")
        self.assertEqual(terminal["step"], 25)
        self.assertEqual(terminal["checkpoint_boundary"], boundary)
        self.assertIsNone(terminal["result"])
        checkpoint = next(
            value
            for value in self.service.checkpoint_rows()
            if value["checkpoint_id"] == boundary["checkpoint_id"]
        )
        self.assertEqual(checkpoint["step"], 25)
        self.__class__.held_checkpoint_id = str(boundary["checkpoint_id"])
        self.s2_observation = {
            "job_id": terminal["job_id"],
            "states_before_cancel": before_cancel,
            "states_after_cancel": after_cancel,
            "state": terminal["state"],
            "terminal_reason": terminal["terminal_reason"],
            "step": terminal["step"],
            "checkpoint_boundary": boundary,
            "checkpoint_manifest_sha256": checkpoint["sha256"],
        }

    def test_s2_native_021_resume_held_checkpoint_to_fifty(self) -> None:
        self.require_full_training()
        self.require_held_checkpoint()
        job, states = self.service.run_job(
            {
                "operation": "tiny_resume",
                "checkpoint_id": self.held_checkpoint_id,
                "additional_steps": 25,
            },
            timeout=600,
        )
        result = self.result(job)
        self.assertEqual(result["operation"], "tiny_resume")
        self.assertEqual(result["parent_checkpoint_id"], self.held_checkpoint_id)
        self.assertEqual(result["completed_step"], 50)
        self.assertEqual(result["requested_final_step"], 50)
        final = next(
            value
            for value in self.service.checkpoint_rows()
            if value["checkpoint_id"] == result["last_checkpoint_id"]
        )
        self.assertEqual(final["step"], 50)
        self.assertEqual(final["parent_checkpoint_id"], self.held_checkpoint_id)
        files = self.service.checkpoint_files(str(final["checkpoint_id"]))
        self.assertEqual(set(files), CHECKPOINT_FILES)
        self.s2_observation = {
            "job_id": job["job_id"],
            "states": states,
            "parent_checkpoint_id": self.held_checkpoint_id,
            "last_checkpoint_id": result["last_checkpoint_id"],
            "completed_step": result["completed_step"],
            "requested_final_step": result["requested_final_step"],
            "checkpoint_files": sorted(files),
        }

    def test_s2_native_025_cli_api_parity_without_dynamic_execution(self) -> None:
        from llm_foundations_companion.scheduler import WorkerController

        sentinel = self.service.root.parent / "learner-text-must-not-execute"
        prompt = f"$(touch {sentinel})"
        request = {
            "operation": "context_preview",
            "backend": "tiny",
            "checkpoint_id": self.full_checkpoint_id,
            "prompt": prompt,
            "max_new_tokens": 4,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 17,
        }
        request_path = self.service.root.parent / "cli-api-request.json"
        request_path.write_bytes(canonical(request))
        key = str(uuid.uuid4())
        api_record = self.service.submit(request, key=key)
        cli_record, argv = self.service.cli_request(request_path, key=key)
        self.assertEqual(cli_record, api_record)
        self.assertEqual(
            api_record["request"]["canonical_sha256"], digest(canonical(request))
        )
        completed, states = self.service.wait_for(
            str(api_record["job_id"]),
            lambda value: value["state"] in TERMINAL,
        )
        self.assertEqual(completed["state"], "completed")
        result = self.result(completed)
        preview = json.loads(
            self.service.artifact_bytes(str(result["preview_artifact_id"]))
        )
        self.assertEqual(preview["tokenizer_sha256"], BYTE_TOKENIZER_SHA256)
        self.assertEqual(prompt.encode("utf-8")[-8:], b"execute)")
        expected_preview = {
            "original_input_token_count": len(prompt.encode("utf-8")),
            "input_token_ids": [101, 120, 101, 99, 117, 116, 101, 41],
            "input_token_count": 8,
            "serialized_text": "execute)",
            "cropped_input_tokens": len(prompt.encode("utf-8")) - 8,
            "effective_context_budget": 8,
        }
        self.assertEqual(
            {key: preview[key] for key in expected_preview}, expected_preview
        )
        self.assertEqual(
            {key: result[key] for key in expected_preview}, expected_preview
        )
        self.assertFalse(sentinel.exists())
        self.assertTrue(all(prompt not in argument for argument in argv))
        self.assertEqual(
            WorkerController().command_prefix,
            (
                sys.executable,
                "-I",
                "-m",
                "llm_foundations_companion.worker_main",
            ),
        )

        spec = importlib.util.find_spec("llm_foundations_companion")
        assert spec is not None and spec.origin is not None
        package = Path(spec.origin).resolve().parent
        offenders = []
        for path in sorted(package.rglob("*.py")):
            tree = ast.parse(path.read_bytes(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if isinstance(function, ast.Name) and function.id in {
                    "eval",
                    "exec",
                    "__import__",
                }:
                    offenders.append(f"{path.name}:{node.lineno}:{function.id}")
                if (
                    isinstance(function, ast.Attribute)
                    and function.attr == "import_module"
                ):
                    offenders.append(
                        f"{path.name}:{node.lineno}:import_module"
                    )
                if any(
                    keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in node.keywords
                ):
                    offenders.append(f"{path.name}:{node.lineno}:shell=True")
        self.assertEqual(offenders, [])
        self.s2_observation = {
            "job_id": completed["job_id"],
            "states": states,
            "idempotency_key": key,
            "canonical_request_sha256": api_record["request"][
                "canonical_sha256"
            ],
            "api_cli_records_equal": True,
            "learner_text_in_argv": False,
            "sentinel_created": False,
            "worker_command_prefix": list(WorkerController().command_prefix),
            "installed_python_files_scanned": len(list(package.rglob("*.py"))),
            "dynamic_execution_offenders": offenders,
        }

    def test_support_training_event_caps_and_full_metrics(self) -> None:
        self.require_full_training()
        job, states = self.service.run_job(
            self.tiny_request(
                dataset_id=str(self.dataset["dataset_id"]),
                tokenizer_id=str(self.byte_tokenizer["tokenizer_id"]),
                steps=40,
                eval_every=1,
            ),
            timeout=600,
        )
        result = self.result(job)
        self.assertEqual(result["completed_step"], 40)
        events = self.service.job_events(str(job["job_id"]))
        counts = Counter(str(event["event_type"]) for event in events)
        self.assertLessEqual(counts["phase_changed"], 32)
        self.assertLessEqual(counts["progress"], 16)
        self.assertEqual(counts["metric"], 41)
        self.assertEqual(counts["checkpoint_committed"], 41)
        self.assertEqual(counts["terminal"], 1)
        metric_events = [
            event for event in events if event["event_type"] == "metric"
        ]
        self.assertEqual(
            [event["payload"]["step"] for event in metric_events],
            list(range(41)),
        )
        self.assertEqual(
            [event["payload"]["name"] for event in metric_events],
            ["validation_nll_token"] * 41,
        )

        checkpoints = sorted(
            self.service.checkpoint_rows(run_id=str(result["run_id"])),
            key=lambda value: value["step"],
        )
        self.assertEqual(
            [int(value["step"]) for value in checkpoints],
            list(range(41)),
        )
        metrics_artifact_id = str(result["metrics_artifact_id"])
        first_raw = self.service.artifact_bytes(metrics_artifact_id)
        metrics = self.metric_rows(first_raw)
        expected_metrics = [
            (
                0,
                "train_nll_token",
                "nats_per_token",
                "tiny-v2-fixed-eight-batches-v1",
            ),
            (
                0,
                "validation_nll_token",
                "nats_per_token",
                "tiny-v2-fixed-eight-batches-v1",
            ),
        ]
        for step in range(1, 41):
            expected_metrics.extend(
                [
                    (
                        step,
                        "train_nll_token",
                        "nats_per_token",
                        "tiny-v2-training-v1",
                    ),
                    (
                        step,
                        "gradient_l2_norm",
                        "l2_norm",
                        "tiny-v2-training-v1",
                    ),
                    (
                        step,
                        "elapsed_seconds",
                        "seconds",
                        "tiny-v2-training-v1",
                    ),
                    (
                        step,
                        "train_nll_token",
                        "nats_per_token",
                        "tiny-v2-fixed-eight-batches-v1",
                    ),
                    (
                        step,
                        "validation_nll_token",
                        "nats_per_token",
                        "tiny-v2-fixed-eight-batches-v1",
                    ),
                ]
            )
        required_metric_keys = {
            "run_id",
            "sequence",
            "step",
            "name",
            "value",
            "unit",
            "protocol_id",
            "recorded_at",
        }
        self.assertEqual(len(metrics), 202)
        self.assertEqual(
            [int(row["sequence"]) for row in metrics], list(range(202))
        )
        self.assertEqual(
            [
                (
                    int(row["step"]),
                    row["name"],
                    row["unit"],
                    row["protocol_id"],
                )
                for row in metrics
            ],
            expected_metrics,
        )
        self.assertTrue(
            all(set(row) == required_metric_keys for row in metrics)
        )
        self.assertTrue(
            all(row["run_id"] == result["run_id"] for row in metrics)
        )
        self.assertTrue(
            all(
                isinstance(row["value"], (int, float))
                and not isinstance(row["value"], bool)
                and math.isfinite(float(row["value"]))
                for row in metrics
            )
        )
        recorded_at_values = {str(row["recorded_at"]) for row in metrics}
        self.assertEqual(len(recorded_at_values), 1)
        descriptor = self.service.get(
            f"/api/v1/artifacts/{metrics_artifact_id}"
        )
        self.assertEqual(descriptor["sha256"], digest(first_raw))
        second_raw = self.service.artifact_bytes(metrics_artifact_id)
        self.assertEqual(second_raw, first_raw)
        self.s2_observation = {
            "job_id": job["job_id"],
            "run_id": result["run_id"],
            "states": states,
            "event_counts": dict(sorted(counts.items())),
            "checkpoint_steps": [int(value["step"]) for value in checkpoints],
            "metrics_rows": len(metrics),
            "metrics_steps": [int(row["step"]) for row in metrics],
            "metric_sequences": [int(row["sequence"]) for row in metrics],
            "metric_names": [str(row["name"]) for row in metrics],
            "metric_units": [str(row["unit"]) for row in metrics],
            "metric_protocol_ids": [
                str(row["protocol_id"]) for row in metrics
            ],
            "metric_recorded_at_values": sorted(recorded_at_values),
            "metrics_artifact_id": metrics_artifact_id,
            "metrics_sha256": digest(first_raw),
            "metrics_immutable_readback": True,
        }
