"""Exact INT-007 limits exposed by ``RuntimeInfo.limits``."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping


LIMITS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "json_body_bytes": 1_048_576,
        "dataset_file_bytes": 10_485_760,
        "dataset_total_bytes": 31_457_280,
        "queued_jobs": 8,
        "compute_seconds": 3_600,
        "max_root_bytes": 53_687_091_200,
        "max_module_keys": 24,
        "max_live_notes": 1_000,
        "max_evidence": 5_000,
        "max_assessments_per_evidence": 10,
        "max_live_conversations": 200,
        "max_turns_per_conversation": 200,
        "max_conversation_text_bytes": 1_048_576,
        "max_capstone_attempts": 100,
        "max_active_previews": 20,
        "max_datasets": 1_000,
        "max_jobs": 50_000,
        "max_runs": 20_000,
        "max_models": 512,
        "max_checkpoints": 5_000,
        "max_artifacts": 100_000,
        "max_events_per_job": 4_096,
        "text_preview_bytes": 262_144,
        "bundle_archive_bytes": 1_073_741_824,
        "bundle_expanded_bytes": 2_147_483_648,
        "tiny_training_text_bytes": 200_000,
        "sft_text_bytes": 20_971_520,
        "dataset_records": 100_000,
        "tiny_updates_per_job": 2_000,
        "adapter_updates_per_lineage": 120,
        "tiny_prompt_bytes": 16_384,
        "tiny_new_tokens": 512,
        "chat_messages": 64,
        "chat_message_bytes": 8_192,
        "chat_total_bytes": 32_768,
        "chat_new_tokens": 128,
        "applied_context_tokens": 512,
        "retrieval_query_bytes": 4_096,
        "retrieval_top_k": 20,
        "backup_wrapper_bytes": 8_388_608,
        "progress_import_bytes": 1_048_576,
        "list_page_items": 100,
        "event_page_items": 500,
        "worker_stream_bytes": 1_048_576,
        "free_space_reserve_bytes": 1_073_741_824,
        "teaching_hold_seconds": 600,
        "cancel_grace_seconds": 30,
        "max_sessions": 32,
        "legacy_backup_bytes": 25_000_000,
        "max_exercise_records": 10_000,
    }
)
