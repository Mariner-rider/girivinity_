from __future__ import annotations
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import psycopg2
import psycopg2.extras
import psycopg2.pool
import yaml

logger = logging.getLogger(__name__)

_POOL: psycopg2.pool.ThreadedConnectionPool | None = None
_POOL_LOCK = threading.Lock()


def _cfg() -> dict:
    return yaml.safe_load(Path("config.yaml").read_text())["database"]


def _password(cfg: dict) -> str:
    # Never store the real password in config.yaml (it's committed to git).
    # DATABASE_PASSWORD env var takes precedence; config.yaml may still hold
    # a value for local/dev convenience, but production should always set
    # the env var instead.
    password = os.environ.get("DATABASE_PASSWORD") or cfg.get("password")
    if not password:
        raise RuntimeError(
            "Database password not configured. Set the DATABASE_PASSWORD "
            "environment variable (see .env.example) or add 'password' "
            "under 'database:' in config.yaml for local development."
        )
    return password


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            cfg = _cfg()
            _POOL = psycopg2.pool.ThreadedConnectionPool(
                minconn=int(cfg.get("pool_min_conn", 2)),
                maxconn=int(cfg.get("pool_max_conn", 10)),
                host=cfg["host"],
                port=int(cfg["port"]),
                dbname=cfg["name"],
                user=cfg["user"],
                password=_password(cfg),
            )
            logger.info("PostgreSQL pool created → %s:%s/%s", cfg["host"], cfg["port"], cfg["name"])
    return _POOL


@contextmanager
def get_conn() -> Generator[psycopg2.extensions.connection, None, None]:
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def execute(sql: str, params: tuple = ()) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)


def executemany(sql: str, params_list: list[tuple]) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, params_list)


def fetchone(sql: str, params: tuple = ()) -> tuple | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def fetchall(sql: str, params: tuple = ()) -> list[tuple]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def close_pool() -> None:
    global _POOL
    with _POOL_LOCK:
        if _POOL:
            _POOL.closeall()
            _POOL = None
            logger.info("PostgreSQL pool closed")
