"""Configuration dataclasses for PAL."""

from dataclasses import dataclass, field
from typing import Tuple

import torch


def default_model_num_workers() -> int:
    """DataLoader workers when not overridden: 8 if CUDA is available, else 0.

    ``train.py`` re-applies auto from ``--device`` when ``--num_workers=-1`` so CPU runs
    on a CUDA-capable machine still use 0 workers by default.
    """
    return 8 if torch.cuda.is_available() else 0


@dataclass
class DataConfig:
    n_compounds: int = 5000
    seed: int = 42


@dataclass
class ModelConfig:
    in_features: int = 2048
    hidden_sizes: tuple = (64,)
    dropout: float = 0.3
    out_features: int = 2
    mc_passes: int = 50
    epochs: int = 200
    batch_size: int = 256
    predict_eval_batch_size: int = 4096
    mc_predict_batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-3
    patience: int = 20
    min_epochs: int = 10
    val_fraction: float = 0.2
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5
    # DataLoader workers; train.py overrides from --num_workers (-1 => auto from --device).
    num_workers: int = field(default_factory=default_model_num_workers)


@dataclass
class ALConfig:
    seed_size: int = 20
    batch_size: int = 10
    n_iterations: int = 50
    n_replicates: int = 3
    val_every: int = 1
    ref_point: Tuple[float, float] = (0.0, 0.0)


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    al: ALConfig = field(default_factory=ALConfig)
    ecfp_radius: int = 2
    ecfp_nbits: int = 2048
    output_dir: str = "pal_results"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    obj_names: Tuple[str, str] = ("Objective 0", "Objective 1")
