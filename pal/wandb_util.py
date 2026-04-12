"""Optional Weights & Biases logging helpers (no hard dependency at import time)."""

from __future__ import annotations

import logging
from typing import Any, Mapping

_log = logging.getLogger("pal")


def wandb_log(data: Mapping[str, Any], *, commit: bool | None = None) -> None:
    """Log metrics if ``wandb.run`` is active; no-op otherwise."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    payload = {k: v for k, v in data.items() if v is not None}
    if not payload:
        return
    try:
        wandb.log(payload, commit=commit)
    except Exception:
        _log.debug("wandb.log failed", exc_info=True)


def strategy_k_param(acq_fn: Any) -> float | None:
    """Best-effort UCB / ellipse k for logging (``None`` if not applicable)."""
    k = getattr(acq_fn, "k_ucb", None)
    if k is not None:
        return float(k)
    k = getattr(acq_fn, "k", None)
    if k is not None:
        return float(k)
    return None
