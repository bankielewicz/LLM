"""S2 operation custody, tokenizer, and checkpoint-ledger schema."""

from .v001 import CORE_STATEMENTS as V001_STATEMENTS


CORE_SCHEMA_VERSION = 2

S2_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS job_operation_contexts (
        job_id TEXT PRIMARY KEY,
        operation TEXT NOT NULL,
        request_sha256 TEXT NOT NULL CHECK (
            length(request_sha256) = 64 AND request_sha256 = lower(request_sha256)
        ),
        runtime_profile TEXT NOT NULL CHECK (
            runtime_profile IN ('win-cpu', 'win-cuda', 'wsl-cpu', 'wsl-cuda')
        ),
        device TEXT NOT NULL CHECK (device IN ('cpu', 'cuda')),
        dependency_lock_sha256 TEXT NOT NULL CHECK (
            length(dependency_lock_sha256) = 64
            AND dependency_lock_sha256 = lower(dependency_lock_sha256)
        ),
        companion_source_revision TEXT NOT NULL CHECK (
            length(companion_source_revision) = 40
            AND companion_source_revision = lower(companion_source_revision)
        ),
        run_id TEXT,
        model_id TEXT,
        tokenizer_id TEXT,
        checkpoint_ids_json TEXT NOT NULL,
        reservation_json TEXT NOT NULL,
        resolved_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS job_operation_contexts_no_update
    BEFORE UPDATE ON job_operation_contexts
    BEGIN SELECT RAISE(ABORT, 'job operation contexts are immutable'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS job_operation_contexts_no_delete
    BEFORE DELETE ON job_operation_contexts
    BEGIN SELECT RAISE(ABORT, 'job operation contexts are immutable'); END
    """,
    """
    CREATE TABLE IF NOT EXISTS tokenizers (
        tokenizer_id TEXT PRIMARY KEY,
        tokenizer_type TEXT NOT NULL CHECK (tokenizer_type IN ('byte', 'byte_bpe')),
        vocab_size INTEGER NOT NULL CHECK (vocab_size BETWEEN 257 AND 1024),
        tokenizer_sha256 TEXT NOT NULL CHECK (
            length(tokenizer_sha256) = 64 AND tokenizer_sha256 = lower(tokenizer_sha256)
        ),
        artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
        job_id TEXT NOT NULL,
        origin TEXT NOT NULL CHECK (
            origin IN ('locally_created', 'imported', 'legacy_imported', 'shipped_fixture')
        ),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT,
        record_json TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS tokenizers_created_idx
    ON tokenizers(created_at DESC, tokenizer_id DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS tokenizers_fingerprint_idx
    ON tokenizers(tokenizer_sha256)
    """,
    """
    CREATE TABLE IF NOT EXISTS checkpoint_files (
        checkpoint_id TEXT NOT NULL REFERENCES checkpoints(checkpoint_id) ON DELETE RESTRICT,
        file_name TEXT NOT NULL,
        artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
        sha256 TEXT NOT NULL CHECK (length(sha256) = 64 AND sha256 = lower(sha256)),
        size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
        PRIMARY KEY(checkpoint_id, file_name),
        UNIQUE(checkpoint_id, artifact_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS checkpoint_files_artifact_idx
    ON checkpoint_files(artifact_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS run_artifacts (
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
        role TEXT NOT NULL,
        artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
        PRIMARY KEY(run_id, role),
        UNIQUE(run_id, artifact_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS run_artifacts_artifact_idx
    ON run_artifacts(artifact_id)
    """,
)

CORE_STATEMENTS = V001_STATEMENTS + S2_STATEMENTS

__all__ = ["CORE_SCHEMA_VERSION", "CORE_STATEMENTS", "S2_STATEMENTS"]
