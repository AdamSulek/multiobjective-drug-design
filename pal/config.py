"""Configuration dataclasses for PAL."""

from dataclasses import dataclass, field
from typing import Tuple


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
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-3
    patience: int = 20
    min_epochs: int = 10
    val_fraction: float = 0.2
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5


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
    device: str = "cpu"
    obj_names: Tuple[str, str] = ("Objective 0", "Objective 1")
