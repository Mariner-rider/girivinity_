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
