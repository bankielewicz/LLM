"""Initial storage, artifact, dataset, capacity, identity, and audit schema."""

CORE_SCHEMA_VERSION = 1

CORE_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS migrations (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL,
        description TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS schema_components (
        component TEXT PRIMARY KEY,
        version INTEGER NOT NULL CHECK (version >= 1),
        installed_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS registry_state (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        database_revision INTEGER NOT NULL CHECK (database_revision >= 0)
    )
    """,
    """
    INSERT OR IGNORE INTO registry_state(singleton, database_revision)
    VALUES (1, 0)
    """,
    """
    CREATE TABLE IF NOT EXISTS root_metadata (
        key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mutation_audit (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        occurred_at TEXT NOT NULL,
        request_id TEXT NOT NULL,
        session_hash TEXT NOT NULL,
        operation TEXT NOT NULL,
        entity_ids_json TEXT NOT NULL,
        outcome TEXT NOT NULL
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS mutation_audit_no_update
    BEFORE UPDATE ON mutation_audit
    BEGIN SELECT RAISE(ABORT, 'mutation audit is append-only'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS mutation_audit_no_delete
    BEFORE DELETE ON mutation_audit
    BEGIN SELECT RAISE(ABORT, 'mutation audit is append-only'); END
    """,
    """
    CREATE TABLE IF NOT EXISTS objects (
        sha256 TEXT PRIMARY KEY CHECK (
            length(sha256) = 64 AND sha256 = lower(sha256)
        ),
        size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
        relative_path TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        display_name TEXT NOT NULL,
        relative_path TEXT NOT NULL,
        size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
        sha256 TEXT NOT NULL REFERENCES objects(sha256) ON DELETE RESTRICT,
        created_at TEXT NOT NULL,
        media_type TEXT NOT NULL,
        preview_policy TEXT NOT NULL CHECK (
            preview_policy IN ('text', 'metadata_only', 'download_only')
        ),
        origin TEXT NOT NULL CHECK (
            origin IN ('locally_created', 'imported', 'legacy_imported', 'shipped_fixture')
        ),
        job_id TEXT,
        content_revision INTEGER NOT NULL DEFAULT 1 CHECK (content_revision = 1),
        updated_at TEXT NOT NULL,
        deleted_at TEXT,
        CHECK (relative_path LIKE 'objects/%')
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS artifacts_created_idx
    ON artifacts(created_at DESC, artifact_id DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS artifacts_job_idx
    ON artifacts(job_id, created_at DESC, artifact_id DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id TEXT PRIMARY KEY,
        format TEXT NOT NULL CHECK (format = 'llm-foundations-dataset-v1'),
        name TEXT NOT NULL,
        record_format TEXT NOT NULL CHECK (
            record_format IN ('document_text_v1', 'instruction_intent_v1', 'retrieval_v1')
        ),
        origin TEXT NOT NULL CHECK (
            origin IN ('locally_created', 'imported', 'legacy_imported', 'shipped_fixture')
        ),
        audit_artifact_id TEXT NOT NULL
            REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
        created_at TEXT NOT NULL,
        provenance_note TEXT,
        manifest_sha256 TEXT NOT NULL,
        eligibility TEXT NOT NULL CHECK (eligibility IN ('eligible', 'audit_only')),
        manifest_json TEXT NOT NULL,
        source_identity_json TEXT,
        content_revision INTEGER NOT NULL DEFAULT 1 CHECK (content_revision = 1),
        updated_at TEXT NOT NULL,
        deleted_at TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS datasets_created_idx
    ON datasets(created_at DESC, dataset_id DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS datasets_manifest_idx
    ON datasets(manifest_sha256)
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_splits (
        dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE RESTRICT,
        split_name TEXT NOT NULL CHECK (split_name IN ('train', 'validation', 'test')),
        artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
        sha256 TEXT NOT NULL,
        records INTEGER NOT NULL CHECK (records BETWEEN 1 AND 100000),
        utf8_bytes INTEGER NOT NULL CHECK (utf8_bytes BETWEEN 1 AND 10485760),
        sealed INTEGER NOT NULL CHECK (sealed IN (0, 1)),
        PRIMARY KEY(dataset_id, split_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        format TEXT NOT NULL,
        operation TEXT NOT NULL,
        origin TEXT NOT NULL CHECK (
            origin IN ('locally_created', 'imported', 'legacy_imported', 'shipped_fixture')
        ),
        job_id TEXT,
        source_identity_json TEXT,
        content_revision INTEGER NOT NULL DEFAULT 1 CHECK (content_revision = 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT,
        record_json TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS runs_created_idx
    ON runs(created_at DESC, run_id DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS runs_operation_idx
    ON runs(operation, created_at DESC, run_id DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS models (
        model_id TEXT PRIMARY KEY,
        format TEXT NOT NULL,
        backend TEXT NOT NULL,
        origin TEXT NOT NULL CHECK (
            origin IN ('locally_created', 'imported', 'legacy_imported', 'shipped_fixture')
        ),
        creating_job_id TEXT,
        source_identity_json TEXT,
        content_revision INTEGER NOT NULL DEFAULT 1 CHECK (content_revision = 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT,
        record_json TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS models_created_idx
    ON models(created_at DESC, model_id DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS checkpoints (
        checkpoint_id TEXT PRIMARY KEY,
        format TEXT NOT NULL,
        backend TEXT NOT NULL,
        origin TEXT NOT NULL CHECK (
            origin IN ('locally_created', 'imported', 'legacy_imported', 'shipped_fixture')
        ),
        run_id TEXT REFERENCES runs(run_id) ON DELETE RESTRICT,
        job_id TEXT,
        manifest_sha256 TEXT NOT NULL,
        source_identity_json TEXT,
        content_revision INTEGER NOT NULL DEFAULT 1 CHECK (content_revision = 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT,
        record_json TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS checkpoints_created_idx
    ON checkpoints(created_at DESC, checkpoint_id DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS lineage_edges (
        source_type TEXT NOT NULL,
        source_id TEXT NOT NULL,
        target_type TEXT NOT NULL,
        target_id TEXT NOT NULL,
        edge_type TEXT NOT NULL,
        creating_job_id TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY(source_type, source_id, target_type, target_id, edge_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS reservations (
        reservation_id TEXT PRIMARY KEY,
        owner_kind TEXT NOT NULL CHECK (owner_kind IN ('job', 'upload')),
        owner_id TEXT NOT NULL,
        byte_count INTEGER NOT NULL CHECK (byte_count >= 0),
        artifact_rows INTEGER NOT NULL CHECK (artifact_rows >= 0),
        dataset_rows INTEGER NOT NULL CHECK (dataset_rows >= 0),
        run_rows INTEGER NOT NULL CHECK (run_rows >= 0),
        model_rows INTEGER NOT NULL CHECK (model_rows >= 0),
        checkpoint_rows INTEGER NOT NULL CHECK (checkpoint_rows >= 0),
        state TEXT NOT NULL CHECK (state IN ('held', 'released')),
        created_at TEXT NOT NULL,
        released_at TEXT,
        UNIQUE(owner_kind, owner_id),
        CHECK (
            (state = 'held' AND released_at IS NULL) OR
            (state = 'released' AND released_at IS NOT NULL)
        )
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS reservations_state_idx ON reservations(state)
    """,
    """
    CREATE INDEX IF NOT EXISTS reservations_owner_idx
    ON reservations(owner_kind, owner_id)
    """,
)
