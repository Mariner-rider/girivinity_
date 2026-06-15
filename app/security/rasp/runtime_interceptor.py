from __future__ import annotations

import builtins
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_ORIGINAL_EVAL = builtins.eval
_ORIGINAL_EXEC = builtins.exec
_ORIGINAL_IMPORT = builtins.__import__

_BLOCKED_MODULES = {"subprocess", "os.system", "pty", "socket_raw", "ctypes", "mmap"}

_INTERCEPTOR_ACTIVE = False
_INTERCEPT_LOG: list[dict] = []


def _safe_eval(source, *args, **kwargs):
    _log_intercept("eval", str(source)[:100])
    if len(str(source)) > 500:
        logger.critical("RASP: eval() blocked — suspicious length: %d", len(str(source)))
        raise PermissionError("RASP: eval() blocked — potential code injection")
    return _ORIGINAL_EVAL(source, *args, **kwargs)


def _safe_exec(source, *args, **kwargs):
    _log_intercept("exec", str(source)[:100])
    src_str = str(source)
    dangerous = ["os.system", "subprocess", "__import__", "socket.connect", "open('/etc", "rm -rf"]
    for d in dangerous:
        if d in src_str:
            logger.critical("RASP: exec() blocked — dangerous pattern: %s", d)
            raise PermissionError(f"RASP: exec() blocked — dangerous pattern: {d}")
    return _ORIGINAL_EXEC(source, *args, **kwargs)


def _safe_import(name, *args, **kwargs):
    if name in _BLOCKED_MODULES:
        logger.warning("RASP: import of '%s' monitored", name)
        _log_intercept("import", name)
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


def _log_intercept(itype: str, detail: str) -> None:
    _INTERCEPT_LOG.append({"type": itype, "detail": detail, "timestamp": datetime.now(timezone.utc).isoformat()})
    if len(_INTERCEPT_LOG) > 1000:
        _INTERCEPT_LOG.pop(0)


class RuntimeInterceptor:
    def activate(self) -> None:
        global _INTERCEPTOR_ACTIVE
        if _INTERCEPTOR_ACTIVE:
            return
        builtins.eval = _safe_eval
        builtins.exec = _safe_exec
        builtins.__import__ = _safe_import
        _INTERCEPTOR_ACTIVE = True
        logger.info("RASP RuntimeInterceptor activated")

    def deactivate(self) -> None:
        global _INTERCEPTOR_ACTIVE
        builtins.eval = _ORIGINAL_EVAL
        builtins.exec = _ORIGINAL_EXEC
        builtins.__import__ = _ORIGINAL_IMPORT
        _INTERCEPTOR_ACTIVE = False
        logger.info("RASP RuntimeInterceptor deactivated")

    def get_intercept_log(self) -> list[dict]:
        return list(_INTERCEPT_LOG[-50:])

    @property
    def is_active(self) -> bool:
        return _INTERCEPTOR_ACTIVE
