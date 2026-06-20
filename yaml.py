"""Small fallback subset of PyYAML used in tests when PyYAML is unavailable."""
from __future__ import annotations
import ast, json, re
from typing import Any

def safe_load(stream: Any) -> Any:
    text = stream.read() if hasattr(stream, 'read') else str(stream)
    try:
        return json.loads(text)
    except Exception:
        pass
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(' ')); stripped = line.strip()
        while stack and indent <= stack[-1][0]: stack.pop()
        parent = stack[-1][1]
        if stripped.startswith('- '):
            val = _parse(stripped[2:].strip())
            if isinstance(parent, list): parent.append(val)
            continue
        if ':' in stripped:
            key, val = stripped.split(':', 1); key = key.strip(); val = val.strip()
            if val == '':
                nxt: Any = [] if _next_is_list(text, raw) else {}
                if isinstance(parent, dict): parent[key] = nxt
                stack.append((indent, nxt))
            else:
                if isinstance(parent, dict): parent[key] = _parse(val)
    return root

def safe_dump(data: Any, *args: Any, **kwargs: Any) -> str:
    return json.dumps(data, indent=2)

def _parse(val: str) -> Any:
    if val in {'true','True'}: return True
    if val in {'false','False'}: return False
    if val in {'null','None','~'}: return None
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    if val.startswith('[') or val.startswith('{'):
        try: return ast.literal_eval(val)
        except Exception: return val
    try: return int(val)
    except Exception: pass
    try: return float(val)
    except Exception: return val

def _next_is_list(text: str, raw: str) -> bool:
    lines = text.splitlines(); idx = lines.index(raw)
    base = len(raw) - len(raw.lstrip(' '))
    for nxt in lines[idx+1:]:
        clean = nxt.split('#',1)[0].rstrip()
        if not clean.strip(): continue
        return (len(clean)-len(clean.lstrip(' ')) > base) and clean.strip().startswith('- ')
    return False

def dump(data: Any, *args: Any, **kwargs: Any) -> str:
    return safe_dump(data, *args, **kwargs)
