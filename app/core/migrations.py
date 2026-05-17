from __future__ import annotations
import logging
from app.core.db import execute

logger = logging.getLogger(__name__)

MIGRATIONS: list[str] = [
"""
    CREATE TABLE IF NOT EXISTS training_queue (
        id          BIGSERIAL PRIMARY KEY,
        query       TEXT        NOT NULL,
        chunk_text  TEXT        NOT NULL,
        url         TEXT        DEFAULT '',
        score       REAL        DEFAULT 0.0,
        timestamp   TIMESTAMPTZ DEFAULT NOW(),
        status      TEXT        DEFAULT 'pending'
    )
""",
"""
    CREATE INDEX IF NOT EXISTS idx_training_queue_status
    ON training_queue (status)
""",
"""
    CREATE TABLE IF NOT EXISTS feedback (
        id          BIGSERIAL PRIMARY KEY,
        user_id     TEXT        NOT NULL,
        score       REAL        NOT NULL,
        timestamp   TIMESTAMPTZ DEFAULT NOW()
    )
""",
"""
    CREATE TABLE IF NOT EXISTS skills (
        slug         TEXT PRIMARY KEY,
        topic        TEXT        NOT NULL,
        version      INTEGER     DEFAULT 1,
        confidence   REAL        DEFAULT 0.5,
        usage_count  INTEGER     DEFAULT 0,
        avg_feedback REAL        DEFAULT 0.0,
        updated_at   TIMESTAMPTZ DEFAULT NOW()
    )
""",
"""
    CREATE TABLE IF NOT EXISTS skill_feedback (
        id          BIGSERIAL PRIMARY KEY,
        slug        TEXT        NOT NULL REFERENCES skills(slug)
                                ON DELETE CASCADE,
        score       REAL        NOT NULL,
        timestamp   TIMESTAMPTZ DEFAULT NOW()
    )
""",
"""
    CREATE INDEX IF NOT EXISTS idx_skill_feedback_slug
    ON skill_feedback (slug)
""",
"""
    CREATE TABLE IF NOT EXISTS skill_interactions (
        id             BIGSERIAL PRIMARY KEY,
        skill_slug     TEXT        NOT NULL,
        query          TEXT        NOT NULL,
        response       TEXT        NOT NULL,
        feedback_score REAL,
        timestamp      TIMESTAMPTZ DEFAULT NOW()
    )
""",
"""
    CREATE INDEX IF NOT EXISTS idx_skill_interactions_slug
    ON skill_interactions (skill_slug)
""",
"""
    CREATE TABLE IF NOT EXISTS sentiment_history (
        id              BIGSERIAL PRIMARY KEY,
        user_id         TEXT        NOT NULL,
        query_hash      TEXT        NOT NULL,
        emotion         TEXT        NOT NULL,
        intensity       REAL        DEFAULT 0.0,
        tone            TEXT        DEFAULT 'neutral',
        language_mix    TEXT        DEFAULT 'english',
        urgency         REAL        DEFAULT 0.0,
        expertise_signal TEXT       DEFAULT 'intermediate',
        response_style  TEXT        DEFAULT 'balanced',
        timestamp       TIMESTAMPTZ DEFAULT NOW()
    )
""",
"""
    CREATE INDEX IF NOT EXISTS idx_sentiment_user
    ON sentiment_history (user_id)
""",
"""
    CREATE TABLE IF NOT EXISTS security_events (
        id          BIGSERIAL PRIMARY KEY,
        user_id     TEXT        DEFAULT 'anonymous',
        ip_address  TEXT        NOT NULL,
        endpoint    TEXT        DEFAULT '',
        event_type  TEXT        NOT NULL,
        threat_type TEXT        DEFAULT 'clean',
        severity    TEXT        DEFAULT 'none',
        detail      TEXT        DEFAULT '',
        blocked     BOOLEAN     DEFAULT FALSE,
        hour_of_day INTEGER,
        timestamp   TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_security_events_user
    ON security_events (user_id, timestamp DESC)
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_security_events_ip
    ON security_events (ip_address, timestamp DESC)
    """,

    """
    CREATE TABLE IF NOT EXISTS rate_limit_buckets (
        id           BIGSERIAL PRIMARY KEY,
        bucket_key   TEXT        NOT NULL,
        request_time TIMESTAMPTZ NOT NULL
    )
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_rate_limit_key_time
    ON rate_limit_buckets (bucket_key, request_time DESC)
    """,

    """
    CREATE TABLE IF NOT EXISTS active_sessions (
        id          BIGSERIAL PRIMARY KEY,
        token       TEXT        UNIQUE NOT NULL,
        user_id     TEXT        NOT NULL,
        ip_address  TEXT        DEFAULT '',
        api_key     TEXT        DEFAULT '',
        created_at  TIMESTAMPTZ DEFAULT NOW(),
        expires_at  TIMESTAMPTZ NOT NULL,
        valid       BOOLEAN     DEFAULT TRUE
    )
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_active_sessions_token
    ON active_sessions (token) WHERE valid = TRUE
    """,

    """
    CREATE TABLE IF NOT EXISTS system_security_mode (
        id           BIGSERIAL PRIMARY KEY,
        mode         TEXT        NOT NULL DEFAULT 'observe',
        triggered_by TEXT        DEFAULT 'system',
        set_at       TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS blacklisted_urls (
        id             BIGSERIAL PRIMARY KEY,
        url            TEXT        NOT NULL,
        url_hash       TEXT        UNIQUE NOT NULL,
        reason         TEXT        DEFAULT '',
        blacklisted_at TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_blacklisted_url_hash
    ON blacklisted_urls (url_hash)
    """,

    """
    CREATE TABLE IF NOT EXISTS tenant_security_configs (
        id                      BIGSERIAL PRIMARY KEY,
        api_key                 TEXT        UNIQUE NOT NULL,
        observe_threshold       REAL        DEFAULT 0.3,
        guard_threshold         REAL        DEFAULT 0.6,
        contain_threshold       REAL        DEFAULT 0.9,
        rate_limit_rpm          INTEGER     DEFAULT 500,
        block_prompt_injection  BOOLEAN     DEFAULT TRUE,
        block_sql_injection     BOOLEAN     DEFAULT TRUE,
        block_xss               BOOLEAN     DEFAULT TRUE,
        block_ssrf              BOOLEAN     DEFAULT TRUE,
        alert_email             TEXT        DEFAULT '',
        custom_blocked_patterns JSONB       DEFAULT '[]',
        updated_at              TIMESTAMPTZ DEFAULT NOW()
    )
    """,
]


def run_migrations() -> None:
    logger.info("Running database migrations...")
    for sql in MIGRATIONS:
        try:
            execute(sql.strip())
        except Exception as exc:
            logger.error("Migration failed: %s\nSQL: %s", exc, sql[:80])
            raise
    logger.info("All migrations complete")
