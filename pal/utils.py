# pal/utils.py
from __future__ import annotations
import os
import sys
import logging
from pathlib import Path
import numpy as np


def seed_everything(seed: int) -> None:
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def setup_logging(output_dir: str) -> logging.Logger:
    logger = logging.getLogger("pal")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(output_dir, exist_ok=True)

    fh = logging.FileHandler(os.path.join(output_dir, "compare.log"), mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def load_seed_indices(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Seed indices file not found: {path}")

    if p.suffix.lower() == ".npy":
        return np.array(np.load(p), dtype=int).ravel()

    text = p.read_text().strip()
    if not text:
        raise ValueError(f"Empty seed indices file: {path}")

    tokens = text.replace(",", " ").split()
    return np.array([int(t) for t in tokens], dtype=int).ravel()