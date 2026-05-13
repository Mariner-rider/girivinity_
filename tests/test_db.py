from unittest.mock import MagicMock, patch


def test_get_pool_creates_pool():
    mock_pool = MagicMock()
    with patch("psycopg2.pool.ThreadedConnectionPool", return_value=mock_pool):
        with patch("app.core.db._POOL", None):
            with patch("yaml.safe_load", return_value={"database": {"host": "localhost", "port": 5432, "name": "test", "user": "u", "password": "p", "pool_min_conn": 1, "pool_max_conn": 5}}):
                from app.core import db as dbmod
                dbmod._POOL = None
                pool = dbmod.get_pool()
    assert pool is not None


def test_fetchone_returns_row():
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_cur.fetchone.return_value = (42,)
    mock_conn.cursor.return_value = mock_cur
    with patch("app.core.db.get_conn") as mock_get:
        mock_get.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get.return_value.__exit__ = MagicMock(return_value=False)
        from app.core.db import fetchone
        result = fetchone("SELECT %s", (42,))
    assert result == (42,)


def test_run_migrations_calls_execute():
    with patch("app.core.migrations.execute") as mock_exec:
        from app.core.migrations import run_migrations
        run_migrations()
        assert mock_exec.call_count >= 5
