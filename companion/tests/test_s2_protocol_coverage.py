from __future__ import annotations

import copy
import hashlib
import io
import os
import struct
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_foundations_companion.schema import canonical_json
from llm_foundations_companion.worker_context import CommitRejected, WorkerContext, WorkerInterrupted
from llm_foundations_companion.worker_protocol import (
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    CancellationToken,
    ProtocolState,
    WorkerProtocolError,
    encode_frame,
    open_inherited_stream,
    read_frame,
    validate_event_payload,
    validate_input_snapshot,
    validate_job_error,
    validate_output_allocations,
    validate_parent_message,
    validate_request_envelope,
    validate_worker_message,
    write_frame,
)

JOB_ID = "123e4567-e89b-42d3-a456-426614174000"
INSTANCE_ID = "123e4567-e89b-42d3-a456-426614174001"
RUN_ID = "123e4567-e89b-42d3-a456-426614174002"
MODEL_ID = "123e4567-e89b-42d3-a456-426614174003"
TOKENIZER_ID = "123e4567-e89b-42d3-a456-426614174004"
CHECKPOINT_ID = "123e4567-e89b-42d3-a456-426614174005"
ARTIFACT_ID = "123e4567-e89b-42d3-a456-426614174006"
NONCE = "a" * 64
DIGEST = "b" * 64
REQUEST = {
    "operation": "tokenizer_train",
    "dataset_id": ARTIFACT_ID,
    "tokenizer_profile_id": "byte-v1",
    "vocab_size": 257,
    "seed": 17,
}
REQUEST_DIGEST = hashlib.sha256(canonical_json(REQUEST)).hexdigest()
ALLOCATIONS = {
    "run_id": RUN_ID,
    "model_id": None,
    "tokenizer_id": TOKENIZER_ID,
    "checkpoint_ids": [],
}
CHECKPOINT_NAMES = (
    "model.safetensors",
    "optimizer.safetensors",
    "rng.safetensors",
    "tokenizer.json",
    "config.json",
    "trainer_state.json",
    "manifest.json",
)


def snapshot(**changes):
    value = {
        "format": "llm-foundations-worker-input-v1",
        "job_id": JOB_ID,
        "operation": "tokenizer_train",
        "request_sha256": REQUEST_DIGEST,
        "runtime_profile": "wsl-cpu",
        "device": "cpu",
        "dependency_lock_sha256": "c" * 64,
        "companion_source_revision": "d" * 40,
        "ids": copy.deepcopy(ALLOCATIONS),
        "resolved": {},
        "inputs": [],
    }
    value.update(changes)
    return value


def envelope():
    snap = snapshot()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "job_id": JOB_ID,
        "instance_id": INSTANCE_ID,
        "spawn_nonce": NONCE,
        "schema_id": "TokenizerTrainRequest",
        "request_sha256": REQUEST_DIGEST,
        "request": copy.deepcopy(REQUEST),
        "input_snapshot_sha256": hashlib.sha256(canonical_json(snap)).hexdigest(),
        "input_snapshot": snap,
        "output_allocations": copy.deepcopy(ALLOCATIONS),
    }


def ready():
    item = envelope()
    return {
        "type": "ready",
        "protocol_version": PROTOCOL_VERSION,
        "job_id": JOB_ID,
        "instance_id": INSTANCE_ID,
        "spawn_nonce": NONCE,
        "request_sha256": REQUEST_DIGEST,
        "schema_id": "TokenizerTrainRequest",
        "input_snapshot_sha256": item["input_snapshot_sha256"],
    }


def job_error(code="WORKER_PROTOCOL_ERROR", retryable=False, field_errors=None):
    return {
        "code": code,
        "message": "The worker rejected unsafe protocol data.",
        "retryable": retryable,
        "field_errors": [] if field_errors is None else field_errors,
    }


def artifact_proposal():
    return {
        "type": "artifact_ready",
        "role": "tokenizer_json",
        "staging_name": "tokenizer.json",
        "size_bytes": 2,
        "sha256": hashlib.sha256(b"{}").hexdigest(),
    }


def checkpoint_proposal(digest=DIGEST):
    return {
        "type": "checkpoint_ready",
        "staging_name": "checkpoint",
        "manifest_sha256": digest,
        "files": [
            {"name": name, "size": 1, "sha256": digest}
            for name in CHECKPOINT_NAMES
        ],
    }


def checkpoint_ack():
    return {
        "type": "checkpoint_committed",
        "checkpoint_id": CHECKPOINT_ID,
        "step": 25,
        "sha256": DIGEST,
        "run_id": RUN_ID,
        "artifact_ids": {
            name: str(uuid.uuid5(uuid.NAMESPACE_URL, name))
            for name in CHECKPOINT_NAMES
        },
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_id", 7, "lowercase UUID"),
        ("run_id", "123E4567-E89B-42D3-A456-426614174002", "lowercase UUID"),
        ("checkpoint_ids", (), "checkpoint_ids"),
        ("checkpoint_ids", [CHECKPOINT_ID, CHECKPOINT_ID], "duplicate"),
        ("checkpoint_ids", [CHECKPOINT_ID] * 2003, "checkpoint_ids"),
    ],
)
def test_allocations_reject_ambiguous_or_unbounded_identities(field, value, message):
    allocations = copy.deepcopy(ALLOCATIONS)
    allocations[field] = value
    with pytest.raises(WorkerProtocolError, match=message):
        validate_output_allocations(allocations)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", "future", "format"),
        ("operation", "learner.module", "operation"),
        ("runtime_profile", "host", "profile"),
        ("device", "tpu", "device"),
        ("companion_source_revision", "x" * 40, "source_revision"),
        ("resolved", [], "resolved"),
        ("inputs", (), "inputs"),
    ],
)
def test_snapshot_rejects_unknown_runtime_or_mutable_shape(field, value, message):
    with pytest.raises(WorkerProtocolError, match=message):
        validate_input_snapshot(snapshot(**{field: value}))


def test_snapshot_rejects_duplicate_roles_and_nonconfined_paths():
    item = {
        "role": "dataset.train",
        "artifact_id": ARTIFACT_ID,
        "sha256": DIGEST,
        "size_bytes": 4,
        "path": "inputs/00-dataset.train",
    }
    with pytest.raises(WorkerProtocolError, match="roles must be unique"):
        validate_input_snapshot(
            snapshot(inputs=[item, {**item, "path": "inputs/01-dataset.train"}])
        )
    with pytest.raises(WorkerProtocolError, match="fixed relative"):
        validate_input_snapshot(snapshot(inputs=[{**item, "path": "../outside"}]))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "before a frame"),
        (b"\x00\x00", "frame header"),
        (struct.pack(">I", 0), "length"),
        (struct.pack(">I", MAX_FRAME_BYTES + 1), "length"),
        (struct.pack(">I", 4) + b"{bad", "strict JSON"),
        (struct.pack(">I", 2) + b"[]", "JSON object"),
        (struct.pack(">I", 4) + b"{}", "strict JSON"),
    ],
)
def test_framing_rejects_truncation_invalid_lengths_and_nonobjects(payload, message):
    with pytest.raises((EOFError, WorkerProtocolError), match=message):
        read_frame(io.BytesIO(payload))


def test_framing_enforces_bound_and_distinguishes_clean_eof():
    with pytest.raises(WorkerProtocolError, match="one MiB"):
        encode_frame({"value": "x" * MAX_FRAME_BYTES})
    assert read_frame(io.BytesIO(), allow_eof=True) is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_version", 2, "unsupported"),
        ("job_id", MODEL_ID, "job identity"),
        ("instance_id", MODEL_ID, "instance identity"),
        ("request", [], "request must be"),
        ("request_sha256", DIGEST, "request digest"),
        ("input_snapshot_sha256", DIGEST, "snapshot digest"),
    ],
)
def test_envelope_rejects_cross_process_identity_and_digest_drift(field, value, message):
    item = envelope()
    item[field] = value
    with pytest.raises(WorkerProtocolError, match=message):
        validate_request_envelope(item, job_id=JOB_ID, instance_id=INSTANCE_ID)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("job_id", MODEL_ID, "snapshot job identity"),
        ("operation", "tiny_train", "snapshot operation"),
        ("request_sha256", DIGEST, "snapshot request digest"),
    ],
)
def test_envelope_rejects_inner_snapshot_binding_drift(field, value, message):
    item = envelope()
    item["input_snapshot"][field] = value
    item["input_snapshot_sha256"] = hashlib.sha256(
        canonical_json(item["input_snapshot"])
    ).hexdigest()
    with pytest.raises(WorkerProtocolError, match=message):
        validate_request_envelope(item, job_id=JOB_ID, instance_id=INSTANCE_ID)


def test_envelope_rejects_allocation_drift_from_snapshot():
    item = envelope()
    item["input_snapshot"]["ids"]["run_id"] = MODEL_ID
    item["input_snapshot_sha256"] = hashlib.sha256(
        canonical_json(item["input_snapshot"])
    ).hexdigest()
    with pytest.raises(WorkerProtocolError, match="allocations do not match"):
        validate_request_envelope(item, job_id=JOB_ID, instance_id=INSTANCE_ID)


@pytest.mark.parametrize(
    ("event_type", "payload", "message"),
    [
        ("future", {}, "unknown worker event"),
        ("state_changed", {"state": "lost", "step": None, "requested_final_step": None}, "unknown job state"),
        ("state_changed", {"state": "running", "step": True, "requested_final_step": None}, "integer bounds"),
        ("phase_changed", {"phase": "thinking"}, "unknown job phase"),
        ("progress", {"current": 2, "total": 1, "unit": "updates", "message": ""}, "exceeds total"),
        ("progress", {"current": 1, "total": 1, "unit": "epochs", "message": ""}, "progress unit"),
        ("metric", {"phase": "eval", "step": 0, "split": "holdout", "name": "loss", "value": 1.0}, "metric split"),
        ("metric", {"phase": "eval", "step": 0, "split": "test", "name": "loss", "value": float("nan")}, "finite"),
        ("warning", {"code": "", "message": "warning"}, "text bounds"),
        ("terminal", {"state": "running", "reason_code": "completed", "checkpoint_id": None, "checkpoint_step": None}, "nonterminal"),
        ("terminal", {"state": "failed", "reason_code": "mystery", "checkpoint_id": None, "checkpoint_step": None}, "closed vocabulary"),
        ("terminal", {"state": "completed", "reason_code": "worker_lost", "checkpoint_id": None, "checkpoint_step": None}, "wrong reason"),
        ("terminal", {"state": "interrupted", "reason_code": "worker_lost", "checkpoint_id": CHECKPOINT_ID, "checkpoint_step": None}, "paired"),
    ],
)
def test_event_contract_rejects_impossible_progress_metric_and_terminal_states(
    event_type, payload, message
):
    with pytest.raises(WorkerProtocolError, match=message):
        validate_event_payload(event_type, payload)


def test_event_contract_accepts_durable_interrupted_boundary():
    value = validate_event_payload(
        "terminal",
        {
            "state": "interrupted",
            "reason_code": "user_cancelled",
            "checkpoint_id": CHECKPOINT_ID,
            "checkpoint_step": 25,
        },
    )
    assert value["checkpoint_step"] == 25


@pytest.mark.parametrize(
    "event_type",
    ["state_changed", "checkpoint_committed", "terminal"],
)
def test_worker_cannot_emit_scheduler_owned_durable_events(event_type):
    payloads = {
        "state_changed": {"state": "running", "step": 0, "requested_final_step": None},
        "checkpoint_committed": {"checkpoint_id": CHECKPOINT_ID, "step": 25, "sha256": DIGEST},
        "terminal": {"state": "completed", "reason_code": "completed", "checkpoint_id": None, "checkpoint_step": None},
    }
    with pytest.raises(WorkerProtocolError, match="scheduler owns"):
        validate_worker_message(
            {"type": "event", "event_type": event_type, "payload": payloads[event_type]}
        )


@pytest.mark.parametrize(
    ("message", "error"),
    [
        ({"type": "artifact_ready", "role": "uploaded_code", "staging_name": "x", "size_bytes": 1, "sha256": DIGEST}, "closed vocabulary"),
        ({"type": "artifact_ready", "role": "run_result", "staging_name": "../x", "size_bytes": 1, "sha256": DIGEST}, "portable basename"),
        ({"type": "artifact_ready", "role": "run_result", "staging_name": "x", "size_bytes": True, "sha256": DIGEST}, "integer bounds"),
        ({"type": "result", "operation": "learner.module", "result": {}}, "operation-discriminated"),
        ({"type": "result", "operation": "generate", "result": []}, "operation-discriminated"),
        ({"type": "interrupted", "reason_code": "completed", "checkpoint_id": None, "checkpoint_step": None, "error": None}, "interruption reason"),
        ({"type": "interrupted", "reason_code": "timeout", "checkpoint_id": CHECKPOINT_ID, "checkpoint_step": None, "error": None}, "paired"),
        ({"type": "unknown"}, "unknown worker message"),
    ],
)
def test_worker_messages_reject_ambiguous_output_or_terminal_claims(message, error):
    with pytest.raises(WorkerProtocolError, match=error):
        validate_worker_message(message)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda item: item.update(files=item["files"][:-1]), "exactly seven"),
        (lambda item: item["files"].__setitem__(1, dict(item["files"][0])), "unique"),
        (lambda item: item["files"][0].update(name="wrong.bin"), "wrong file set"),
        (lambda item: item.update(manifest_sha256="c" * 64), "manifest digest"),
    ],
)
def test_checkpoint_proposals_reject_incomplete_or_inconsistent_membership(mutate, error):
    item = checkpoint_proposal()
    mutate(item)
    with pytest.raises(WorkerProtocolError, match=error):
        validate_worker_message(item)


def test_interrupted_message_accepts_only_narrow_checkpoint_failure_error():
    message = {
        "type": "interrupted",
        "reason_code": "user_cancelled",
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_step": 25,
        "error": job_error("CHECKPOINT_WRITE_FAILED", retryable=True),
    }
    assert validate_worker_message(message) == message
    message["error"] = job_error("WORKER_PROTOCOL_ERROR")
    with pytest.raises(WorkerProtocolError, match="closed vocabulary"):
        validate_worker_message(message)


@pytest.mark.parametrize(
    ("message", "error"),
    [
        ({"type": "artifact_prepared", "role": "uploaded_code", "artifact_id": ARTIFACT_ID, "sha256": DIGEST}, "role is invalid"),
        ({"type": "checkpoint_committed", **{k: v for k, v in checkpoint_ack().items() if k != "type"}, "artifact_ids": {}}, "artifact_ids is invalid"),
        ({"type": "checkpoint_committed", **{k: v for k, v in checkpoint_ack().items() if k != "type"}, "artifact_ids": {**checkpoint_ack()["artifact_ids"], "extra": ARTIFACT_ID}}, "artifact_ids is invalid"),
        ({"type": "commit_rejected", "kind": "result", "error": job_error()}, "kind is invalid"),
        ({"type": "unknown"}, "unknown parent message"),
    ],
)
def test_parent_acknowledgments_reject_unbound_or_unknown_claims(message, error):
    with pytest.raises(WorkerProtocolError, match=error):
        validate_parent_message(message)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"code": "NOT_A_CODE"}, "closed vocabulary"),
        ({"retryable": 1}, "boolean"),
        ({"field_errors": {}}, "field_errors"),
        ({"field_errors": [{"field_path": "", "message": "bad"}]}, "text bounds"),
        ({"field_errors": [{"field_path": "request.seed", "message": ""}]}, "text bounds"),
    ],
)
def test_job_error_rejects_open_codes_and_malformed_field_details(change, error):
    item = job_error()
    item.update(change)
    with pytest.raises(WorkerProtocolError, match=error):
        validate_job_error(item)


def protocol_state():
    return ProtocolState(
        JOB_ID,
        INSTANCE_ID,
        NONCE,
        REQUEST_DIGEST,
        "TokenizerTrainRequest",
        "tokenizer_train",
        envelope()["input_snapshot_sha256"],
    )


def test_protocol_state_rejects_identity_drift_duplicate_ready_and_unprompted_ack():
    state = protocol_state()
    drifted = ready()
    drifted["spawn_nonce"] = "c" * 64
    with pytest.raises(WorkerProtocolError, match="identity"):
        state.accept(drifted)
    state.accept(ready())
    with pytest.raises(WorkerProtocolError, match="more than once"):
        state.accept(ready())
    with pytest.raises(WorkerProtocolError, match="no pending proposal"):
        state.acknowledge(
            {"type": "artifact_prepared", "role": "run_result", "artifact_id": ARTIFACT_ID, "sha256": DIGEST}
        )


def test_protocol_state_binds_each_ack_kind_and_terminal_ordering():
    state = protocol_state()
    state.accept(ready())
    state.accept(checkpoint_proposal())
    with pytest.raises(WorkerProtocolError, match="kind does not match"):
        state.acknowledge(
            {"type": "artifact_prepared", "role": "run_result", "artifact_id": ARTIFACT_ID, "sha256": DIGEST}
        )
    rejected = {"type": "commit_rejected", "kind": "checkpoint", "error": job_error()}
    assert state.acknowledge(rejected) == rejected
    interrupted = {"type": "interrupted", "reason_code": "timeout", "checkpoint_id": None, "checkpoint_step": None, "error": None}
    assert state.accept(interrupted) == interrupted
    with pytest.raises(WorkerProtocolError, match="after a terminal"):
        state.accept({"type": "error", "error": job_error()})


@pytest.mark.parametrize("reason", ["user_cancelled", "timeout", "service_shutdown"])
def test_cancellation_token_accepts_one_closed_reason(reason):
    stream = io.BytesIO()
    write_frame(stream, {"reason_code": reason})
    stream.seek(0)
    token = CancellationToken(stream)
    assert token.wait(1)
    assert token.cancelled is True
    assert token.reason == reason


@pytest.mark.parametrize(
    "payload",
    [{"reason_code": "worker_lost"}, {"reason_code": "timeout", "extra": 1}],
)
def test_cancellation_token_ignores_unrecognized_or_malformed_requests(payload):
    stream = io.BytesIO()
    write_frame(stream, payload)
    stream.seek(0)
    token = CancellationToken(stream)
    assert token.wait(0.1) is False
    assert token.cancelled is False
    assert token.reason is None


def test_open_inherited_stream_consumes_only_a_valid_nonstdio_descriptor(monkeypatch):
    monkeypatch.delenv("S2_PIPE", raising=False)
    with pytest.raises(WorkerProtocolError, match="missing inherited"):
        open_inherited_stream("S2_PIPE", "rb")
    monkeypatch.setenv("S2_PIPE", "not-a-fd")
    with pytest.raises(WorkerProtocolError, match="invalid inherited"):
        open_inherited_stream("S2_PIPE", "rb")
    monkeypatch.setenv("S2_PIPE", "2")
    with pytest.raises(WorkerProtocolError, match="invalid inherited"):
        open_inherited_stream("S2_PIPE", "rb")
    read_fd, write_fd = os.pipe()
    try:
        monkeypatch.setenv("S2_PIPE", str(read_fd))
        stream = open_inherited_stream("S2_PIPE", "rb")
        os.write(write_fd, b"x")
        assert stream.read(1) == b"x"
        stream.close()
        assert "S2_PIPE" not in os.environ
    finally:
        os.close(write_fd)
