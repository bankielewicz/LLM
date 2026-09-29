#!/usr/bin/env python3
"""Offline, executable edge review for the intermediate-v1 contracts.

This tool validates specification artifacts only. It does not start the product,
open a browser, use the network, or execute model/training code.

Default: write a fresh Markdown report and return nonzero for failed checks or findings.
Existing reports are never overwritten. --allow-findings records a known-open attempt.
--stdout also prints the generated report.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator, FormatChecker


BASE = Path(__file__).resolve().parents[1]
CONTRACTS = BASE / "contracts"
SCHEMAS = CONTRACTS / "schemas"
FIXTURES = BASE / "fixtures"
DEFAULT_REPORT = BASE / "reviews" / "contract-edge-checks.md"
OPENAPI_PATH = CONTRACTS / "openapi.json"
MODEL_PROFILE_SCHEMA_PATH = CONTRACTS / "model-profile.json"

UUIDS = {
    name: f"{index:08x}-1111-4111-8111-{index:012x}"
    for index, name in enumerate(
        (
            "installation", "dataset", "tokenizer", "model", "checkpoint",
            "parent_checkpoint", "run", "artifact", "metrics", "records",
            "paired", "evaluation", "adapter_run", "best_checkpoint",
            "last_checkpoint", "retrieval", "bundle", "evidence", "assessment",
            "job", "conversation", "generation", "note", "attempt",
            "preview", "verification",
        ),
        start=1,
    )
}
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
WHEN = "2026-09-28T12:00:00Z"
RELEASE_TOKEN = "A" * 43
FORMAT_CHECKER = FormatChecker()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def pointer_get(document: Any, fragment: str) -> Any:
    if fragment in ("", "#"):
        return document
    if not fragment.startswith("#/"):
        raise KeyError(f"unsupported fragment: {fragment}")
    current = document
    for raw_part in fragment[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def schema_key(path: Path) -> str:
    name = path.name
    if name.endswith(".schema.json"):
        name = name[: -len(".schema.json")]
    elif name.endswith(".json"):
        name = name[:-5]
    return "external__" + name


def rewrite_refs(value: Any, local_prefix: str | None = None) -> Any:
    """Rewrite contract refs into one in-memory Draft 2020-12 bundle."""
    if isinstance(value, list):
        return [rewrite_refs(item, local_prefix) for item in value]
    if not isinstance(value, dict):
        return value
    output: dict[str, Any] = {}
    for key, item in value.items():
        if key == "$id":
            continue
        if key != "$ref" or not isinstance(item, str):
            output[key] = rewrite_refs(item, local_prefix)
            continue
        if item.startswith("#/components/schemas/"):
            output[key] = "#/$defs/" + item.rsplit("/", 1)[1]
            continue
        if item.startswith("#"):
            output[key] = (local_prefix + item[1:]) if local_prefix else item
            continue
        target, marker, fragment = item.partition("#")
        candidate = Path(target).name
        if candidate.endswith(".schema.json"):
            rewritten = "#/$defs/" + schema_key(Path(candidate))
            if marker:
                rewritten += fragment
            output[key] = rewritten
            continue
        output[key] = item
    return output


def build_bundle(openapi: dict[str, Any], schema_docs: dict[str, Any]) -> dict[str, Any]:
    definitions = {
        name: rewrite_refs(schema)
        for name, schema in openapi["components"]["schemas"].items()
    }
    for filename, schema in schema_docs.items():
        definition_key = schema_key(Path(filename))
        definitions[definition_key] = rewrite_refs(
            schema, "#/" + chr(36) + f"defs/{definition_key}"
        )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://llm-foundations.local/contracts/intermediate-v1-bundle.json",
        "$defs": definitions,
    }


def validator_for(bundle: dict[str, Any], definition: str) -> Draft202012Validator:
    schema = copy.deepcopy(bundle)
    schema["$ref"] = f"#/$defs/{definition}"
    return Draft202012Validator(schema, format_checker=FORMAT_CHECKER)


def errors_for(
    bundle: dict[str, Any], definition: str, instance: Any
) -> list[str]:
    validator = validator_for(bundle, definition)
    errors = sorted(
        validator.iter_errors(instance),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    return [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: "
        f"{error.message}"
        for error in errors
    ]


def is_valid(bundle: dict[str, Any], definition: str, instance: Any) -> bool:
    return not errors_for(bundle, definition, instance)


def ref_inventory(openapi: dict[str, Any], schema_docs: dict[str, Any]) -> tuple[int, list[str]]:
    failures: list[str] = []
    count = 0

    def walk(value: Any, owner: str, document: Any, owner_path: Path) -> None:
        nonlocal count
        if isinstance(value, list):
            for item in value:
                walk(item, owner, document, owner_path)
            return
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            if key == "$ref" and isinstance(item, str):
                count += 1
                try:
                    if item.startswith("#"):
                        pointer_get(document, item)
                    else:
                        target_text, marker, fragment = item.partition("#")
                        target_path = (owner_path.parent / target_text).resolve()
                        if not target_path.is_file():
                            raise FileNotFoundError(target_path)
                        target_doc = read_json(target_path)
                        if marker:
                            pointer_get(target_doc, "#" + fragment)
                except (KeyError, IndexError, FileNotFoundError, json.JSONDecodeError) as exc:
                    failures.append(f"{owner}: {item}: {exc}")
            else:
                walk(item, owner, document, owner_path)

    walk(openapi, OPENAPI_PATH.name, openapi, OPENAPI_PATH)
    for filename, document in schema_docs.items():
        walk(document, filename, document, SCHEMAS / filename)
    return count, failures


def request_digest(request: dict[str, Any]) -> str:
    unbound = {
        key: value
        for key, value in request.items()
        if key not in {"preview_artifact_id", "context_preview_digest"}
    }
    return sha256_json(unbound)


def preview_matches(preview: dict[str, Any], request: dict[str, Any]) -> bool:
    return (
        preview["preview_artifact_id"] == request.get("preview_artifact_id")
        and preview["context_preview_digest"] == request.get("context_preview_digest")
        and preview["canonical_generation_request_sha256"] == request_digest(request)
    )


class Review:
    def __init__(self) -> None:
        self.checks: list[tuple[str, str, str]] = []
        self.findings: list[tuple[str, str, str, str]] = []

    def check(self, case_id: str, passed: bool, detail: str) -> None:
        self.checks.append((case_id, "PASS" if passed else "FAIL", detail))

    def finding(self, finding_id: str, severity: str, title: str, evidence: str) -> None:
        self.findings.append((finding_id, severity, title, evidence))


def operation_from_ref(reference: str) -> str:
    return reference.rsplit("/", 1)[1]



def build_examples(model_files: list[dict[str, Any]]) -> tuple[
    dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]
]:
    ids = UUIDS
    preview_digest = SHA_B

    generate_request = {
        "operation": "generate",
        "model_id": ids["model"],
        "prompt": "The next token is",
        "max_new_tokens": 16,
        "temperature": 0,
        "top_p": 1,
        "seed": 17,
        "preview_artifact_id": ids["preview"],
        "context_preview_digest": preview_digest,
    }
    chat_request = {
        "operation": "chat_generate",
        "subject": {"model_id": ids["model"]},
        "messages": [{"role": "user", "content": "Classify: reset my password"}],
        "max_new_tokens": 24,
        "temperature": 0,
        "top_p": 1,
        "seed": 17,
        "conversation_id": ids["conversation"],
        "expected_conversation_revision": 1,
        "preview_artifact_id": ids["preview"],
        "context_preview_digest": preview_digest,
    }

    requests: dict[str, dict[str, Any]] = {
        "tokenizer_train": {
            "operation": "tokenizer_train",
            "dataset_id": ids["dataset"],
            "vocab_size": 300,
            "seed": 17,
            "tokenizer_profile_id": "byte-bpe-v1",
        },
        "tiny_train": {
            "operation": "tiny_train",
            "dataset_id": ids["dataset"],
            "tokenizer_id": ids["tokenizer"],
            "device": "cpu",
            "steps": 30,
            "eval_every": 10,
            "batch_size": 4,
            "learning_rate": 0.001,
            "seed": 17,
            "context": 32,
            "width": 64,
            "heads": 4,
            "layers": 2,
            "architecture_profile_id": "tiny-v2-standard-v1",
        },
        "tiny_resume": {
            "operation": "tiny_resume",
            "checkpoint_id": ids["checkpoint"],
            "device": "cpu",
            "additional_steps": 10,
        },
        "evaluate": {
            "operation": "evaluate",
            "subjects": [
                {"kind": "base_model", "model_id": ids["model"]},
                {"kind": "adapter", "checkpoint_id": ids["checkpoint"]},
            ],
            "dataset_id": ids["dataset"],
            "split": "test",
            "evaluation_profile_id": "applied-intents-greedy-v1",
            "release_token": RELEASE_TOKEN,
        },
        "generate": generate_request,
        "model_prepare": {
            "operation": "model_prepare",
            "model_profile_id": "smollm2-135m-instruct-v1",
            "accept_download": True,
        },
        "adapter_train": {
            "operation": "adapter_train",
            "model_id": ids["model"],
            "dataset_id": ids["dataset"],
            "preset_id": "smollm2-intents-lora-v1",
            "seed": 17,
            "updates": 20,
        },
        "adapter_resume": {
            "operation": "adapter_resume",
            "checkpoint_id": ids["checkpoint"],
            "additional_updates": 10,
        },
        "context_preview": {
            "operation": "context_preview",
            "backend": "tiny",
            "model_id": ids["model"],
            "prompt": generate_request["prompt"],
            "max_new_tokens": generate_request["max_new_tokens"],
            "temperature": generate_request["temperature"],
            "top_p": generate_request["top_p"],
            "seed": generate_request["seed"],
        },
        "chat_generate": chat_request,
        "retrieval_build": {
            "operation": "retrieval_build",
            "dataset_id": ids["dataset"],
            "split": "train",
            "text_field": "text",
            "max_features": 1000,
        },
        "retrieval_query": {
            "operation": "retrieval_query",
            "retrieval_index_id": ids["retrieval"],
            "query": "reset password",
            "top_k": 3,
        },
        "export_bundle": {
            "operation": "export_bundle",
            "bundle_mode": "full",
            "run_ids": [ids["run"]],
            "model_ids": [ids["model"]],
            "checkpoint_options": [
                {
                    "checkpoint_id": ids["checkpoint"],
                    "include_resume_state": True,
                }
            ],
            "evidence_ids": [ids["evidence"]],
            "note_ids": [ids["note"]],
            "conversation_ids": [ids["conversation"]],
        },
        "validate_bundle": {
            "operation": "validate_bundle",
            "bundle_artifact_id": ids["bundle"],
        },
        "import_bundle": {
            "operation": "import_bundle",
            "bundle_artifact_id": ids["bundle"],
            "duplicate_policy": "skip",
        },
    }

    artifact_ids = [ids["artifact"]]
    results: dict[str, dict[str, Any]] = {
        "tokenizer_train": {
            "operation": "tokenizer_train",
            "tokenizer_id": ids["tokenizer"],
            "vocab_size": 300,
            "tokenizer_sha256": SHA_A,
            "artifact_ids": artifact_ids,
            "tokenizer_type": "byte_bpe",
        },
        "tiny_train": {
            "operation": "tiny_train",
            "run_id": ids["run"],
            "model_id": ids["model"],
            "checkpoint_id": ids["checkpoint"],
            "completed_step": 30,
            "requested_final_step": 30,
            "artifact_ids": artifact_ids,
        },
        "tiny_resume": {
            "operation": "tiny_resume",
            "run_id": ids["run"],
            "model_id": ids["model"],
            "checkpoint_id": ids["checkpoint"],
            "completed_step": 40,
            "requested_final_step": 40,
            "parent_checkpoint_id": ids["parent_checkpoint"],
            "artifact_ids": artifact_ids,
        },
        "evaluate": {
            "operation": "evaluate",
            "evaluation_id": ids["evaluation"],
            "subjects": requests["evaluate"]["subjects"],
            "metrics_artifact_id": ids["metrics"],
            "records_artifact_id": ids["records"],
            "paired_artifact_id": ids["paired"],
            "artifact_ids": [ids["metrics"], ids["records"], ids["paired"]],
        },
        "generate": {
            "operation": "generate",
            "run_id": ids["run"],
            "model_id": ids["model"],
            "generated_text": " a model.",
            "generated_token_count": 3,
            "stop_reason": "eos",
            "serialized_input": generate_request["prompt"],
            "truncated_input_tokens": 0,
            "artifact_ids": artifact_ids,
        },
        "model_prepare": {
            "operation": "model_prepare",
            "model_id": ids["model"],
            "model_profile_id": "smollm2-135m-instruct-v1",
            "download_manifest_sha256": "4ac640c0c740b5294fbcc1bfe0d74a4667262c98c63ac65d456d12a0648cda0a",
            "total_bytes": 272437573,
            "files": model_files,
            "verification": "verified",
            "artifact_ids": artifact_ids,
        },
        "adapter_train": {
            "operation": "adapter_train",
            "adapter_run_id": ids["adapter_run"],
            "best_checkpoint_id": ids["best_checkpoint"],
            "last_checkpoint_id": ids["last_checkpoint"],
            "completed_update": 20,
            "metrics_artifact_id": ids["metrics"],
            "artifact_ids": artifact_ids,
        },
        "adapter_resume": {
            "operation": "adapter_resume",
            "adapter_run_id": ids["adapter_run"],
            "best_checkpoint_id": ids["best_checkpoint"],
            "last_checkpoint_id": ids["last_checkpoint"],
            "completed_update": 30,
            "metrics_artifact_id": ids["metrics"],
            "parent_checkpoint_id": ids["checkpoint"],
            "artifact_ids": artifact_ids,
        },
        "context_preview": {
            "operation": "context_preview",
            "preview_artifact_id": ids["preview"],
            "backend": "tiny",
            "subject_identity_sha256": SHA_A,
            "tokenizer_sha256": SHA_B,
            "template_sha256": None,
            "device": "cpu",
            "input_token_ids": [1, 2, 3, 4],
            "input_token_count": 4,
            "serialized_text": generate_request["prompt"],
            "dropped_message_indices": [],
            "cropped_input_tokens": 0,
            "effective_context_budget": 496,
            "canonical_generation_request_sha256": request_digest(generate_request),
            "artifact_ids": [ids["preview"]],
            "context_preview_digest": preview_digest,
            "original_input_token_count": 4,
        },
        "chat_generate": {
            "operation": "chat_generate",
            "generation_id": ids["generation"],
            "output_artifact_id": ids["artifact"],
            "text": "account_access",
            "generated_token_count": 2,
            "stop_reason": "eos",
            "artifact_ids": artifact_ids,
        },
        "retrieval_build": {
            "operation": "retrieval_build",
            "retrieval_index_id": ids["retrieval"],
            "dataset_id": ids["dataset"],
            "document_count": 30,
            "vocabulary_size": 600,
            "artifact_ids": artifact_ids,
        },
        "retrieval_query": {
            "operation": "retrieval_query",
            "retrieval_index_id": ids["retrieval"],
            "query": requests["retrieval_query"]["query"],
            "matches": [
                {
                    "record_id": "train-001",
                    "rank": 1,
                    "score": 0.9,
                    "text": "reset password instructions",
                }
            ],
            "artifact_ids": artifact_ids,
        },
        "export_bundle": {
            "operation": "export_bundle",
            "bundle_artifact_id": ids["bundle"],
            "bundle_sha256": SHA_A,
            "included_identity_count": 6,
            "artifact_ids": [ids["bundle"]],
        },
        "validate_bundle": {
            "operation": "validate_bundle",
            "bundle_artifact_id": ids["bundle"],
            "bundle_sha256": SHA_A,
            "valid": True,
            "manifest_version": "1.0",
            "identity_counts": {"artifact": 1},
            "warnings": [],
            "artifact_ids": [ids["bundle"]],
        },
        "import_bundle": {
            "operation": "import_bundle",
            "bundle_sha256": SHA_A,
            "identity_map": [
                {
                    "entity_type": "artifact",
                    "original_id": ids["artifact"],
                    "local_id": None,
                    "disposition": "skipped",
                }
            ],
            "imported_counts": {"artifact": 0},
            "artifact_ids": [],
            "duplicate_source_count": 1,
            "skipped_original_ids": [ids["artifact"]],
        },
    }

    extras = {
        "generate_request": generate_request,
        "chat_request": chat_request,
        "tiny_preview_result": results["context_preview"],
    }
    return requests, results, extras



def runtime_example() -> dict[str, Any]:
    limits = {
        "json_body_bytes": 1048576,
        "dataset_file_bytes": 10485760,
        "dataset_total_bytes": 31457280,
        "queued_jobs": 8,
        "compute_seconds": 3600,
        "max_root_bytes": 53687091200,
        "max_module_keys": 24,
        "max_live_notes": 1000,
        "max_evidence": 5000,
        "max_assessments_per_evidence": 10,
        "max_live_conversations": 200,
        "max_turns_per_conversation": 200,
        "max_conversation_text_bytes": 1048576,
        "max_capstone_attempts": 100,
        "max_active_previews": 20,
        "max_datasets": 1000,
        "max_jobs": 50000,
        "max_runs": 20000,
        "max_models": 512,
        "max_checkpoints": 5000,
        "max_artifacts": 100000,
        "max_events_per_job": 4096,
        "text_preview_bytes": 262144,
        "bundle_archive_bytes": 1073741824,
        "bundle_expanded_bytes": 2147483648,
    }
    return {
        "mode": "local",
        "api_version": "1.0",
        "runtime_version": "1.0.0",
        "instance_id": UUIDS["installation"],
        "profile": "wsl-cpu",
        "device": "cpu",
        "storage_root_display": "Learner storage",
        "storage_writable": True,
        "server_time": WHEN,
        "limits": limits,
        "capabilities": {
            "tiny_train": {"available": True},
            "cuda": {
                "available": False,
                "reason_code": "DEVICE_UNAVAILABLE",
                "message": "CUDA is not available in this profile.",
            },
        },
    }


def base_job(operation: str = "tokenizer_train") -> dict[str, Any]:
    return {
        "job_id": UUIDS["job"],
        "operation": operation,
        "state": "queued",
        "created_at": WHEN,
        "updated_at": WHEN,
        "started_at": None,
        "finished_at": None,
        "progress": {
            "current": 0,
            "total": None,
            "unit": "updates",
            "message": "Queued",
        },
        "result": None,
        "error": None,
        "last_cursor": 1,
        "phase": None,
        "step": None,
        "requested_final_step": None,
        "warnings": [],
        "warning_suppressed_count": 0,
        "checkpoint_boundary": None,
        "terminal_reason": None,
    }


def job_state_examples(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    queued = base_job()
    starting = copy.deepcopy(queued)
    starting.update(
        state="starting",
        started_at=WHEN,
        phase="loading",
        last_cursor=2,
    )
    running = copy.deepcopy(starting)
    running.update(
        state="running",
        phase="training",
        step=1,
        requested_final_step=30,
        last_cursor=3,
    )
    running["progress"] = {
        "current": 1,
        "total": 30,
        "unit": "updates",
        "message": "Training",
    }
    cancelling = copy.deepcopy(running)
    cancelling.update(state="cancelling", last_cursor=4)
    completed = copy.deepcopy(running)
    completed.update(
        state="completed",
        finished_at=WHEN,
        phase=None,
        step=30,
        result=results["tokenizer_train"],
        last_cursor=5,
        terminal_reason="completed",
    )
    failed = copy.deepcopy(starting)
    failed.update(
        state="failed",
        finished_at=WHEN,
        phase=None,
        error={
            "code": "WORKER_PROTOCOL_ERROR",
            "message": "Worker returned an invalid result.",
            "retryable": False,
            "field_errors": [],
        },
        last_cursor=3,
        terminal_reason="worker_protocol_error",
    )
    interrupted = copy.deepcopy(running)
    interrupted.update(
        state="interrupted",
        finished_at=WHEN,
        phase=None,
        step=25,
        last_cursor=6,
        checkpoint_boundary={
            "checkpoint_id": UUIDS["checkpoint"],
            "step": 25,
        },
        terminal_reason="exercise_timeout",
    )
    return {
        "queued": queued,
        "starting": starting,
        "running": running,
        "cancelling": cancelling,
        "completed": completed,
        "failed": failed,
        "interrupted": interrupted,
    }


def assessment_request(rubric: dict[str, Any], review_kind: str) -> dict[str, Any]:
    return {
        "expected_evidence_revision": 1,
        "review_kind": review_kind,
        "reviewer": {
            "reviewer_id": "learner-01" if review_kind == "self" else "reviewer-01",
            "display_name": "Learner" if review_kind == "self" else "Independent reviewer",
        },
        "rubric_version": "intermediate-v1",
        "decision": "meets",
        "rationale": "Artifacts meet the bounded rubric criteria.",
        "rubric_rows": [
            {
                "criterion_id": row["id"],
                "score": 10,
                "comment": "Met with cited evidence.",
            }
            for row in rubric["rows"]
        ],
    }


def stored_assessment(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "assessment_id": UUIDS["assessment"],
        "evidence_id": UUIDS["evidence"],
        "review_kind": request["review_kind"],
        "reviewer": request["reviewer"],
        "rubric_version": request["rubric_version"],
        "rubric_rows": request["rubric_rows"],
        "decision": request["decision"],
        "rationale": request["rationale"],
        "origin": "locally_created",
        "created_at": WHEN,
    }


def capstone_examples() -> tuple[dict[str, Any], dict[str, Any]]:
    create = {
        "reason": "first_attempt",
        "baseline_subject": {"model_id": UUIDS["model"]},
        "candidate_subject": {"checkpoint_id": UUIDS["checkpoint"]},
        "dataset_id": UUIDS["dataset"],
        "dataset_manifest_sha256": SHA_A,
        "evaluation_profile_id": "applied-intents-greedy-v1",
    }
    created = {
        "attempt": {
            "attempt_id": UUIDS["attempt"],
            "reason": "first_attempt",
            "baseline_subject": {
                "kind": "base_model",
                "model_id": UUIDS["model"],
            },
            "candidate_subject": {
                "kind": "adapter",
                "checkpoint_id": UUIDS["checkpoint"],
            },
            "dataset_id": UUIDS["dataset"],
            "dataset_manifest_sha256": SHA_A,
            "evaluation_profile_id": "applied-intents-greedy-v1",
            "prior_release_count": 0,
            "token_sha256": hashlib.sha256(RELEASE_TOKEN.encode("ascii")).hexdigest(),
            "token_state": "available",
            "test_exposure": "fresh_test",
            "created_at": WHEN,
        },
        "release_token": RELEASE_TOKEN,
    }
    return create, created


def progress_example() -> dict[str, Any]:
    module_ids = ["P00"] + [f"{index:02d}" for index in range(21)] + ["E01", "E02"]
    modules = {
        module_id: {
            "reading": module_id == "P00",
            "read": False,
            "practiced": False,
            "self_checked": False,
            "revision": 1,
            "updated_at": WHEN,
        }
        for module_id in module_ids
    }
    modules["P00"]["reading_at"] = WHEN
    return {
        "format": "llm-foundations-progress-v2",
        "edition": "intermediate-v1",
        "installation_id": UUIDS["installation"],
        "revision": 1,
        "modules": modules,
        "updated_at": WHEN,
    }


def learning_state_example(progress: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "llm-foundations-learning-state-v1",
        "schema_version": 1,
        "edition": "intermediate-v1",
        "source": {
            "surface": "static",
            "installation_id": UUIDS["installation"],
            "app_version": "1.0.0",
        },
        "exported_at": WHEN,
        "progress": progress,
        "notes": [],
        "evidence_claims": [],
        "conversations": [],
        "payload_sha256": SHA_A,
    }


def collect_curriculum_ids(value: Any) -> Iterable[str]:
    if isinstance(value, list):
        for item in value:
            yield from collect_curriculum_ids(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in {"module_id", "source_lesson_id", "destination_module_id"}:
                if isinstance(item, str):
                    yield item
            elif key == "prerequisites" and isinstance(item, list):
                yield from (entry for entry in item if isinstance(entry, str))
            else:
                yield from collect_curriculum_ids(item)



def operation_consts(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, list):
        for item in value:
            found.update(operation_consts(item))
    elif isinstance(value, dict):
        operation = value.get("properties", {}).get("operation", {})
        if isinstance(operation, dict) and isinstance(operation.get("const"), str):
            found.add(operation["const"])
        for item in value.values():
            found.update(operation_consts(item))
    return found


def run_review() -> tuple[Review, dict[str, str]]:
    review = Review()
    openapi = read_json(OPENAPI_PATH)
    schema_docs = {
        path.name: read_json(path)
        for path in sorted(SCHEMAS.glob("*.json"))
    }
    bundle = build_bundle(openapi, schema_docs)
    components = openapi["components"]["schemas"]

    compile_failures: list[str] = []
    for filename, schema in schema_docs.items():
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as exc:  # schema diagnostics must be retained
            compile_failures.append(f"{filename}: {exc}")
    try:
        Draft202012Validator.check_schema(read_json(MODEL_PROFILE_SCHEMA_PATH))
    except Exception as exc:
        compile_failures.append(f"{MODEL_PROFILE_SCHEMA_PATH.name}: {exc}")
    try:
        Draft202012Validator.check_schema(bundle)
    except Exception as exc:
        compile_failures.append(f"in-memory OpenAPI schema bundle: {exc}")
    review.check(
        "CEC-SCHEMA-001",
        not compile_failures,
        (
            f"{len(schema_docs) + 2} Draft 2020-12 schema units compile"
            if not compile_failures
            else "; ".join(compile_failures)
        ),
    )

    ref_count, ref_failures = ref_inventory(openapi, schema_docs)
    review.check(
        "CEC-SCHEMA-002",
        not ref_failures,
        (
            f"{ref_count} local/external refs resolve"
            if not ref_failures
            else "; ".join(ref_failures)
        ),
    )

    request_refs = [
        item["$ref"] for item in components["JobRequest"]["oneOf"]
    ]
    result_refs = [
        item["$ref"] for item in components["JobResult"]["oneOf"]
    ]
    request_names = [operation_from_ref(ref) for ref in request_refs]
    result_names = [operation_from_ref(ref) for ref in result_refs]
    request_operations = set().union(
        *(operation_consts(components[name]) for name in request_names)
    )
    result_operations = set().union(
        *(operation_consts(components[name]) for name in result_names)
    )
    job_operations = set(components["Job"]["properties"]["operation"]["enum"])
    expected_operations = {
        "tokenizer_train", "tiny_train", "tiny_resume", "evaluate", "generate",
        "model_prepare", "adapter_train", "adapter_resume", "context_preview",
        "chat_generate", "retrieval_build", "retrieval_query", "export_bundle",
        "validate_bundle", "import_bundle",
    }
    discriminator_ok = (
        request_operations == expected_operations
        and result_operations == expected_operations
        and job_operations == expected_operations
        and len(request_refs) == 15
        and len(result_refs) == 15
    )
    review.check(
        "CEC-API-001",
        discriminator_ok,
        "request, result and Job operation sets are the same closed 15-operation set",
    )

    model_manifest = read_json(
        FIXTURES / "applied" / "model-download-manifest.json"
    )
    requests, results, extras = build_examples(model_manifest["files"])

    for operation in sorted(expected_operations):
        request = requests[operation]
        union_valid = is_valid(bundle, "JobRequest", request)
        branch_count = sum(
            is_valid(bundle, name, request) for name in request_names
        )
        review.check(
            f"CEC-REQ-{operation.upper()}",
            union_valid and branch_count == 1,
            f"normal {operation} request matches JobRequest and exactly one branch "
            f"(branches={branch_count})",
        )

        result = results[operation]
        union_result_valid = is_valid(bundle, "JobResult", result)
        result_branch_count = sum(
            is_valid(bundle, name, result) for name in result_names
        )
        review.check(
            f"CEC-RES-{operation.upper()}",
            union_result_valid and result_branch_count == 1,
            f"normal {operation} result matches JobResult and exactly one branch "
            f"(branches={result_branch_count})",
        )

    runtime = runtime_example()
    review.check(
        "CEC-API-002",
        is_valid(bundle, "RuntimeInfo", runtime),
        "normal GET /api/v1/runtime response validates with every fixed limit",
    )

    state_examples = job_state_examples(results)
    for state, example in state_examples.items():
        review.check(
            f"CEC-STATE-{state.upper()}",
            is_valid(bundle, "Job", example),
            f"normal {state} Job representation validates",
        )

    valid_event = {
        "job_id": UUIDS["job"],
        "cursor": 1,
        "event_type": "state_changed",
        "occurred_at": WHEN,
        "payload": {
            "state": "queued",
            "step": None,
            "requested_final_step": None,
        },
    }
    review.check(
        "CEC-EVENT-001",
        is_valid(bundle, "JobEvent", valid_event),
        "normal state_changed event validates",
    )

    chat_preview_request = {
        "operation": "context_preview",
        "backend": "chat",
        "subject": extras["chat_request"]["subject"],
        "messages": extras["chat_request"]["messages"],
        "max_new_tokens": extras["chat_request"]["max_new_tokens"],
        "temperature": extras["chat_request"]["temperature"],
        "top_p": extras["chat_request"]["top_p"],
        "seed": extras["chat_request"]["seed"],
        "conversation_id": extras["chat_request"]["conversation_id"],
        "expected_conversation_revision": extras["chat_request"][
            "expected_conversation_revision"
        ],
    }
    chat_preview_result = copy.deepcopy(extras["tiny_preview_result"])
    chat_preview_result.update(
        backend="chat",
        template_sha256=SHA_C,
        serialized_text="<user>Classify: reset my password</user>",
        canonical_generation_request_sha256=request_digest(extras["chat_request"]),
    )
    review.check(
        "CEC-PREVIEW-001",
        is_valid(bundle, "ContextPreviewRequest", chat_preview_request)
        and is_valid(bundle, "ContextPreviewResult", chat_preview_result)
        and preview_matches(
            extras["tiny_preview_result"], extras["generate_request"]
        )
        and preview_matches(chat_preview_result, extras["chat_request"]),
        "tiny and chat preview examples validate and bind to their generation requests",
    )

    stale_generate = copy.deepcopy(extras["generate_request"])
    stale_generate["prompt"] = "Changed after preview"
    review.check(
        "CEC-PREVIEW-002",
        is_valid(bundle, "GenerateRequest", stale_generate)
        and not preview_matches(extras["tiny_preview_result"], stale_generate),
        "stale changed request remains shape-valid but the executable digest check rejects it",
    )

    capstone_create, capstone_created = capstone_examples()
    token_matches = (
        hashlib.sha256(
            capstone_created["release_token"].encode("ascii")
        ).hexdigest()
        == capstone_created["attempt"]["token_sha256"]
    )
    review.check(
        "CEC-TOKEN-001",
        is_valid(bundle, "CapstoneAttemptCreateRequest", capstone_create)
        and is_valid(
            bundle,
            schema_key(Path("capstone-attempt-created.schema.json")),
            capstone_created,
        )
        and token_matches
        and requests["evaluate"]["release_token"]
        == capstone_created["release_token"],
        "capstone create/created shapes validate and the one-time token matches its stored SHA-256 and evaluate request",
    )

    rubric = read_json(FIXTURES / "curriculum" / "capstone-rubric.json")
    rubric_ids = [row["id"] for row in rubric["rows"]]
    self_request = assessment_request(rubric, "self")
    independent_request = assessment_request(rubric, "independent")
    review.check(
        "CEC-RUBRIC-001",
        len(rubric_ids) == 10
        and len(set(rubric_ids)) == 10
        and is_valid(bundle, "AssessmentRequest", self_request)
        and is_valid(bundle, "AssessmentRequest", independent_request)
        and is_valid(
            bundle,
            schema_key(Path("assessment-record.schema.json")),
            stored_assessment(independent_request),
        ),
        "10-row rubric and normal self/independent assessment records validate",
    )
    missing_reviewer = copy.deepcopy(self_request)
    missing_reviewer.pop("reviewer")
    review.check(
        "CEC-RUBRIC-002",
        not is_valid(bundle, "AssessmentRequest", missing_reviewer),
        "assessment request without reviewer is rejected",
    )
    invalid_score = copy.deepcopy(self_request)
    invalid_score["rubric_rows"][0]["score"] = 7
    review.check(
        "CEC-RUBRIC-003",
        not is_valid(bundle, "AssessmentRequest", invalid_score),
        "assessment request with a score outside 0/5/10 is rejected",
    )

    progress = progress_example()
    valid_module_ids = set(progress["modules"])
    review.check(
        "CEC-STATIC-001",
        len(valid_module_ids) == 24
        and is_valid(
            bundle,
            schema_key(Path("progress-v2.schema.json")),
            progress,
        ),
        "all 24 static module IDs and independent progress flags validate",
    )
    invalid_progress = read_json(
        FIXTURES / "data" / "invalid-progress-module21.json"
    )
    review.check(
        "CEC-STATIC-002",
        not is_valid(
            bundle,
            schema_key(Path("progress-v2.schema.json")),
            invalid_progress,
        ),
        "out-of-range static module ID 21 is rejected",
    )
    learning_state = learning_state_example(progress)
    review.check(
        "CEC-STATIC-003",
        is_valid(
            bundle,
            schema_key(Path("learning-state-backup.schema.json")),
            learning_state,
        ),
        "static installation UUID, independent progress, and portable learning-state wrapper validate",
    )
    overlay = read_json(FIXTURES / "curriculum" / "curriculum-overlay.json")
    overlay_ids = set(collect_curriculum_ids(overlay))
    review.check(
        "CEC-STATIC-004",
        overlay_ids <= valid_module_ids,
        f"all {len(overlay_ids)} module references in curriculum-overlay.json are recognized",
    )

    validation_cases = read_json(
        FIXTURES / "data" / "validation-cases.json"
    )["cases"]
    fixture_failures: list[str] = []
    for case in validation_cases:
        schema_path = (
            FIXTURES / "data" / case["schema"]
        ).resolve()
        instance_path = FIXTURES / "data" / case["instance"]
        actual = is_valid(bundle, schema_key(schema_path), read_json(instance_path))
        if actual != case["valid"]:
            fixture_failures.append(
                f"{case['id']} expected {case['valid']} got {actual}"
            )
    review.check(
        "CEC-FIXTURE-001",
        not fixture_failures,
        (
            f"all {len(validation_cases)} declared data validation fixtures match expectation"
            if not fixture_failures
            else "; ".join(fixture_failures)
        ),
    )
    model_profile_schema = read_json(MODEL_PROFILE_SCHEMA_PATH)
    applied_profile = read_json(
        FIXTURES / "applied" / "smollm2-135m-instruct-v1.json"
    )
    profile_valid = not list(
        Draft202012Validator(
            model_profile_schema,
            format_checker=FORMAT_CHECKER,
        ).iter_errors(applied_profile)
    )
    review.check(
        "CEC-FIXTURE-002",
        profile_valid,
        "applied pinned model profile validates against model-profile.json",
    )



    supplied_assessment = read_json(
        FIXTURES / "data" / "valid-assessment.json"
    )
    supplied_ids = [
        row["criterion_id"] for row in supplied_assessment["rubric_rows"]
    ]
    if set(supplied_ids) != set(rubric_ids):
        review.finding(
            "CEC-F001",
            "P1",
            "The supplied valid assessment does not identify the shipped rubric rows",
            "valid-assessment.json uses "
            + ", ".join(supplied_ids)
            + "; capstone-rubric.json uses "
            + ", ".join(rubric_ids)
            + ". The generic criterion_id schema accepts both, so the declared "
            "valid fixture can be stored without referring to any actual rubric row.",
        )

    empty_rationale = copy.deepcopy(independent_request)
    empty_rationale["rationale"] = ""
    empty_stored = stored_assessment(empty_rationale)
    empty_request_valid = is_valid(
        bundle, "AssessmentRequest", empty_rationale
    )
    empty_stored_valid = is_valid(
        bundle,
        schema_key(Path("assessment-record.schema.json")),
        empty_stored,
    )
    if empty_request_valid and not empty_stored_valid:
        review.finding(
            "CEC-F002",
            "P1",
            "AssessmentRequest accepts a value the stored assessment contract rejects",
            "OpenAPI AssessmentRequest rationale has minLength 0, while "
            "assessment-record.schema.json requires minLength 1. A normal accepted "
            "empty rationale cannot be represented by the response/storage contract.",
        )

    duplicate_rows = copy.deepcopy(independent_request)
    duplicate_rows["rubric_rows"][1]["criterion_id"] = duplicate_rows[
        "rubric_rows"
    ][0]["criterion_id"]
    duplicate_rows["rubric_rows"][1]["comment"] = "Different comment."
    if is_valid(bundle, "AssessmentRequest", duplicate_rows):
        review.finding(
            "CEC-F003",
            "P1",
            "Assessment schemas allow duplicate and omitted rubric criteria",
            "uniqueItems compares whole row objects. Two rows with the same "
            "criterion_id and different comments validate, so one required rubric "
            "criterion may be omitted while the request still contains 10 rows.",
        )

    invalid_decision = copy.deepcopy(independent_request)
    for row in invalid_decision["rubric_rows"]:
        row["score"] = 0
    invalid_decision["decision"] = "meets"
    if is_valid(bundle, "AssessmentRequest", invalid_decision):
        review.finding(
            "CEC-F004",
            "P1",
            "Assessment decision is not constrained by rubric score and objective gates",
            "A request with ten zero scores and decision meets validates. The shipped "
            "rubric requires objective gates, total at least 80, and no zero rows.",
        )

    impossible_jobs: dict[str, dict[str, Any]] = {}
    impossible_jobs["queued_with_started_at"] = copy.deepcopy(
        state_examples["queued"]
    )
    impossible_jobs["queued_with_started_at"]["started_at"] = WHEN
    impossible_jobs["completed_without_result"] = copy.deepcopy(
        state_examples["completed"]
    )
    impossible_jobs["completed_without_result"]["result"] = None
    impossible_jobs["failed_without_error"] = copy.deepcopy(
        state_examples["failed"]
    )
    impossible_jobs["failed_without_error"]["error"] = None
    accepted_impossible = [
        name
        for name, example in impossible_jobs.items()
        if is_valid(bundle, "Job", example)
    ]
    if accepted_impossible:
        review.finding(
            "CEC-F005",
            "P1",
            "Job schema accepts state combinations forbidden by the runtime contract",
            "Accepted impossible examples: "
            + ", ".join(accepted_impossible)
            + ". Runtime requires queued jobs not to have started and completion to "
            "have a typed result; failed jobs require failure data.",
        )

    mismatched_job = copy.deepcopy(state_examples["completed"])
    mismatched_job["operation"] = "generate"
    if is_valid(bundle, "Job", mismatched_job):
        review.finding(
            "CEC-F006",
            "P1",
            "Job operation is not bound to its typed result discriminator",
            "A completed Job with operation generate and a tokenizer_train result "
            "validates. Runtime treats discriminator mismatch as WORKER_PROTOCOL_ERROR.",
        )

    mismatched_event = copy.deepcopy(valid_event)
    mismatched_event["payload"] = {
        "current": 1,
        "total": 10,
        "unit": "updates",
        "message": "progress",
    }
    if is_valid(bundle, "JobEvent", mismatched_event):
        review.finding(
            "CEC-F007",
            "P1",
            "Job event_type is not bound to its payload variant",
            "A state_changed event carrying ProgressPayload validates because payload "
            "is an uncorrelated oneOf. Consumers cannot safely dispatch on event_type.",
        )

    invalid_tiny = copy.deepcopy(requests["tiny_train"])
    invalid_tiny["steps"] = 5
    invalid_tiny["eval_every"] = 10
    if is_valid(bundle, "TinyTrainRequest", invalid_tiny):
        review.finding(
            "CEC-F008",
            "P2",
            "TinyTrainRequest omits the eval_every no-greater-than-steps invariant",
            "steps=5 and eval_every=10 validates although RUN-011 requires "
            "eval_every no greater than steps before job creation.",
        )

    altered_prepare = copy.deepcopy(results["model_prepare"])
    altered_prepare["files"][0]["bytes"] = 0
    if is_valid(bundle, "ModelPrepareResult", altered_prepare):
        review.finding(
            "CEC-F009",
            "P2",
            "ModelPrepareResult does not bind the fixed file manifest to total_bytes",
            "Changing one authoritative file byte count to zero still validates while "
            "total_bytes remains 272437573. Runtime text requires eight exact files, "
            "sizes, digests, and exact total.",
        )

    source_hashes = {
        "openapi.json": hashlib.sha256(OPENAPI_PATH.read_bytes()).hexdigest(),
        "model-profile.json": hashlib.sha256(
            MODEL_PROFILE_SCHEMA_PATH.read_bytes()
        ).hexdigest(),
        "capstone-rubric.json": hashlib.sha256(
            (FIXTURES / "curriculum" / "capstone-rubric.json").read_bytes()
        ).hexdigest(),
        "schema_count": str(len(schema_docs)),
        "ref_count": str(ref_count),
    }
    return review, source_hashes


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_report(review: Review, source_hashes: dict[str, str]) -> str:
    passed = sum(status == "PASS" for _, status, _ in review.checks)
    failed = len(review.checks) - passed
    lines = [
        "# Intermediate v1 contract edge checks",
        "",
        "This is an offline specification-contract review. It executed Python "
        "`Draft202012Validator` checks and small semantic oracles only. It did "
        "not start the companion, execute product Python, train or run a model, "
        "open a browser, use the network, or assess rendered accessibility.",
        "",
        "## Outcome",
        "",
        f"- Executable checks: **{passed} passed, {failed} failed, "
        f"{len(review.checks)} total**.",
        f"- Open findings: **{len(review.findings)}**.",
        "- Browser, native runtime, installation, training, inference, and "
        "accessibility acceptance: **NOT_RUN**.",
        "",
        "Schema compilation alone is not treated as useful validation. The checks "
        "also exercise every job request/result discriminator, normal runtime/job "
        "states, preview and capstone token binding, assessment edge cases, static "
        "module IDs, declared fixtures, and cross-file references.",
        "",
        "## Source identity",
        "",
        "| Artifact | SHA-256 / value |",
        "|---|---|",
    ]
    for name, value in source_hashes.items():
        lines.append(f"| {name} | `{value}` |")

    lines.extend(
        [
            "",
            "## Open findings",
            "",
        ]
    )
    if review.findings:
        for finding_id, severity, title, evidence in review.findings:
            lines.extend(
                [
                    f"### {finding_id} · {severity} · {title}",
                    "",
                    evidence,
                    "",
                ]
            )
    else:
        lines.extend(["No open contract findings were reproduced.", ""])

    lines.extend(
        [
            "## Executed cases",
            "",
            "| Case | Status | Observation |",
            "|---|---|---|",
        ]
    )
    for case_id, status, detail in review.checks:
        lines.append(
            f"| `{case_id}` | **{status}** | {markdown_escape(detail)} |"
        )

    lines.extend(
        [
            "",
            "## Boundary interpretation",
            "",
            "- `CEC-PREVIEW-002` proves the JSON shape alone cannot detect a "
            "stale preview. The separately executed digest oracle rejects the changed "
            "request, matching RUN-011; the implementation still needs a native test.",
            "- `CEC-TOKEN-001` checks the raw one-time capstone token against the "
            "stored SHA-256 and the sealed evaluate request. It does not prove atomic "
            "consumption or replay rejection.",
            "- The state examples establish representability. They do not prove "
            "transactions, ordering, cancellation, recovery, or worker behavior.",
            "- Static backup checks establish schema compatibility only. They do not "
            "exercise localStorage quotas, browser file pickers, migration, or reload.",
            "",
            "## Reproduction",
            "",
            "From the specification repository root:",
            "",
            "```bash",
            "python3 docs/specs/intermediate-v1/tools/check_contract_edges.py "
            "--report docs/specs/intermediate-v1/reviews/<fresh-name>.md",
            "```",
            "",
            "The tool refuses to overwrite an existing report. It returns nonzero "
            "when any executable check fails or any finding remains open. Use "
            "`--allow-findings` only when intentionally recording a review attempt "
            "with known open findings; the report still lists them.",
            "",
        ]
    )
    return "\n".join(lines)


def next_report_path() -> Path:
    for index in range(1, 1000):
        candidate = BASE / "reviews" / f"contract-edge-attempt{index:03d}.md"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no free contract-edge-attemptNNN.md report name")


def resolve_report_path(raw: str | None) -> Path:
    candidate = next_report_path() if raw is None else Path(raw)
    if not candidate.is_absolute():
        candidate = (BASE / candidate).resolve()
    reviews_root = (BASE / "reviews").resolve()
    if not candidate.resolve().is_relative_to(reviews_root):
        raise ValueError("report path must be inside docs/specs/intermediate-v1/reviews")
    if candidate.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {candidate}")
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        help="fresh report path, relative to intermediate-v1 or absolute",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="also print the generated Markdown report",
    )
    parser.add_argument(
        "--allow-findings",
        action="store_true",
        help="return zero despite retained open findings; failed checks still return nonzero",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="compatibility flag; strict is already the default",
    )
    args = parser.parse_args(argv)

    try:
        report_path = resolve_report_path(args.report)
        review, source_hashes = run_review()
        report = render_report(review, source_hashes)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(report)
        if args.stdout:
            print(report)
        print(report_path)
        failed_checks = any(status == "FAIL" for _, status, _ in review.checks)
        if failed_checks:
            return 1
        if review.findings and not args.allow_findings:
            return 1
        return 0
    except Exception as exc:
        print(f"contract edge review failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
