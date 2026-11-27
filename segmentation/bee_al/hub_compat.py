"""Reuse of the hub's parsing functions, without duplicating them.

``clean_label`` and ``parse_shifted_row_*`` live in ``active_learning_tool/app.py``.
Importing that module would start Flask and create a ``results`` folder in the
current directory, so here only the needed definitions are extracted from the source.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Callable, NamedTuple, Union

from .config import REPO_ROOT

HUB_APP_PATH = REPO_ROOT / "active_learning_tool" / "app.py"
_NAMES = ("clean_label", "parse_shifted_row_single", "parse_shifted_row_grid")


class HubHelpers(NamedTuple):
    clean_label: Callable[[str], str]
    parse_shifted_row_single: Callable[[dict], tuple[str, str]]
    parse_shifted_row_grid: Callable[[dict], tuple[str, str]]


@lru_cache(maxsize=None)
def load_hub_helpers(app_path: Union[str, Path] = HUB_APP_PATH) -> HubHelpers:
    """Compile only the parsing functions taken from ``app.py``."""
    path = Path(app_path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _NAMES]
    missing = set(_NAMES) - {n.name for n in nodes}
    if missing:
        raise ImportError(f"{path} no longer defines: {sorted(missing)}")
    namespace: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return HubHelpers(*(namespace[name] for name in _NAMES))
