"""Process-wide preferences for optional diagnostic / timer logging.

Used by ``src/train.py`` (and any caller that invokes ``set_pal_log_prefs``) so
deep call sites (e.g. ``batch_delta_hv_3d``) can gate ``[DIAG]`` / ``[TIMER]``
without threading flags through every function.

Defaults preserve historical behavior: both enabled when unset.
"""

from __future__ import annotations

import contextvars
from typing import Any, Tuple

_diag: contextvars.ContextVar[bool] = contextvars.ContextVar("pal_log_diag", default=True)
_timer: contextvars.ContextVar[bool] = contextvars.ContextVar("pal_log_timer", default=True)


def set_pal_log_prefs(*, log_diag: bool, log_timer: bool) -> Tuple[Any, Any]:
    """Return tokens; caller must ``reset_pal_log_prefs`` when done (e.g. in ``finally``)."""
    return _diag.set(bool(log_diag)), _timer.set(bool(log_timer))


def reset_pal_log_prefs(tokens: Tuple[Any, Any]) -> None:
    _diag.reset(tokens[0])
    _timer.reset(tokens[1])


def pal_log_diag() -> bool:
    return bool(_diag.get())


def pal_log_timer() -> bool:
    return bool(_timer.get())
