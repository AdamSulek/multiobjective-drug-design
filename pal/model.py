"""MLP with MC-dropout for multi-objective prediction (dimension-agnostic)."""

from __future__ import annotations

import copy
import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig

logger = logging.getLogger(__name__)


def _r2_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Per-target R² for shape (N, d). Returns tensor (d,)."""
    # handle constant targets safely
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    y_mean = torch.mean(y_true, dim=0)
    ss_tot = torch.sum((y_true - y_mean) ** 2, dim=0).clamp_min(1e-12)
    return 1.0 - ss_res / ss_tot


def _log_cuda_mem(prefix: str = "") -> None:
    """Log basic CUDA allocator stats (GB). Safe to call on CPU-only machines."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserv = torch.cuda.memory_reserved() / (1024**3)
        max_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        msg_prefix = (prefix + " ") if prefix else ""
        # logger.info(
        #     "%sCUDA mem: allocated=%.2fGB reserved=%.2fGB max_alloc=%.2fGB",
        #     msg_prefix,
        #     alloc,
        #     reserv,
        #     max_alloc,
        # )


class MLP(nn.Module):
    """Multi-layer perceptron with dropout (shared backbone + linear head)."""

    def __init__(
        self,
        in_features: int = 2048,
        hidden_sizes: tuple[int, ...] = (64,),
        dropout: float = 0.2,
        out_features: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = []

        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, int(out_features))

    @property
    def out_features(self) -> int:
        return int(self.head.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class PiALMultilabelMLP(nn.Module):
    """Shared ECFP encoder with one regression and two multilabel heads."""

    def __init__(
        self,
        in_features: int = 2048,
        hidden_sizes: tuple[int, ...] = (1024, 512, 256),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(in_features)
        for hidden in hidden_sizes:
            layers.extend((nn.Linear(previous, int(hidden)), nn.ReLU()))
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            previous = int(hidden)
        self.encoder = nn.Sequential(*layers)
        self.docking_head = nn.Linear(previous, 1)
        self.selected_head = nn.Linear(previous, 8)
        self.remaining_head = nn.Linear(previous, 48)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        return (
            self.docking_head(embedding),
            self.selected_head(embedding),
            self.remaining_head(embedding),
        )

    def probabilities(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        docking, selected_logits, remaining_logits = self(x)
        return docking, torch.sigmoid(selected_logits), torch.sigmoid(remaining_logits)

    def acquisition_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Return docking and expected selected/remaining counts."""
        docking, selected_probs, remaining_probs = self.probabilities(x)
        return torch.cat(
            (
                docking,
                selected_probs.sum(dim=1, keepdim=True),
                remaining_probs.sum(dim=1, keepdim=True),
            ),
            dim=1,
        )


class MoleculeDataset(Dataset):
    """Simple dataset wrapping numpy feature and label arrays."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(np.asarray(X)).float()
        self.Y = torch.from_numpy(np.asarray(Y)).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


class PiALMultilabelDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        docking: np.ndarray,
        selected: np.ndarray,
        remaining: np.ndarray,
    ):
        n = len(X)
        if np.asarray(selected).shape != (n, 8) or np.asarray(remaining).shape != (n, 48):
            raise ValueError("Expected selected (N,8) and remaining (N,48)")
        self.X = torch.from_numpy(np.asarray(X)).float()
        self.docking = torch.from_numpy(np.asarray(docking).reshape(n, 1)).float()
        self.selected = torch.from_numpy(np.asarray(selected)).float()
        self.remaining = torch.from_numpy(np.asarray(remaining)).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.docking[idx], self.selected[idx], self.remaining[idx]


def build_model(cfg: ModelConfig, device: str = "cpu", out_features: int | None = None) -> MLP:
    """Construct an MLP from config and move to device.

    If out_features is provided, it overrides cfg.out_features.
    """
    of = int(out_features) if out_features is not None else int(getattr(cfg, "out_features", 2))
    model = MLP(
        in_features=int(cfg.in_features),
        hidden_sizes=tuple(cfg.hidden_sizes),
        dropout=float(cfg.dropout),
        out_features=of,
    ).to(device)
    return model


def build_pial_multilabel_model(
    cfg: ModelConfig, device: str = "cpu"
) -> PiALMultilabelMLP:
    """Build the canonical 1-regression + 8/48-multilabel PiAL model."""
    return PiALMultilabelMLP(
        in_features=int(cfg.in_features),
        hidden_sizes=tuple(cfg.hidden_sizes),
        dropout=float(cfg.dropout),
    ).to(device)


def train_model(
    model: MLP,
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    patience: int = 20,
    min_epochs: int = 10,
    val_fraction: float = 0.2,
    lr_scheduler_patience: int = 10,
    lr_scheduler_factor: float = 0.5,
) -> float:
    """Train *model* from scratch on (X, Y) with early stopping.

    When the dataset has >= 10 samples and patience > 0, an internal
    validation split is created and monitored for early stopping.
    Returns the best monitored loss.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Re-initialise weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    n_samples = len(X)
    use_val = n_samples >= 10 and patience > 0

    if use_val:
        rng = np.random.RandomState(n_samples)
        idx = np.arange(n_samples)
        rng.shuffle(idx)

        n_val = max(1, int(n_samples * val_fraction))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        X_tr, Y_tr = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        val_ds = MoleculeDataset(X_val, Y_val)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        X_tr, Y_tr = X, Y
        val_loader = None
        val_ds = None

    ds = MoleculeDataset(X_tr, Y_tr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=lr_scheduler_factor, patience=lr_scheduler_patience
    )
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    wait = 0

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        train_loss = epoch_loss / len(ds)

        if use_val and val_loader is not None and val_ds is not None:
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_trues = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    val_loss += criterion(pred, yb).item() * len(xb)
                    val_preds.append(pred)
                    val_trues.append(yb)

            val_loss /= len(val_ds)

            # R² per objective (d-dim)
            val_pred_all = torch.cat(val_preds, dim=0)
            val_true_all = torch.cat(val_trues, dim=0)
            _ = _r2_torch(val_true_all, val_pred_all)

            model.train()
            monitored = val_loss
        else:
            monitored = train_loss

        scheduler.step(monitored)

        if monitored < best_loss:
            best_loss = monitored
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch >= min_epochs and patience > 0 and wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info(
        "Train done: n=%d use_val=%s best_loss=%.6g epochs_ran<=%d device=%s out_features=%d",
        n_samples,
        use_val,
        best_loss,
        epochs,
        device,
        model.out_features,
    )
    return best_loss


def mc_predict(
    model: MLP,
    X: np.ndarray,
    n_passes: int = 50,
    batch_size: int = 2048,
    device: str = "cpu",
    return_samples: bool = False,
) -> (
    Tuple[np.ndarray, np.ndarray, np.ndarray]
    | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Run MC-dropout inference using batch-first Welford accumulation.

    Keeps dropout active (model.train()) and computes mean/cov per batch.
    Dimension is inferred from the model output (d = model.out_features).

    NOTE: return_samples=True can be huge: allocates (N, n_passes, d) on CPU.
    """
    model.train()  # keep dropout active

    X = np.asarray(X)
    N = len(X)
    d = int(model.out_features)

    if N == 0:
        means = np.zeros((0, d), dtype=np.float32)
        stds = np.zeros((0, d), dtype=np.float32)
        covs = np.zeros((0, d, d), dtype=np.float32)
        if return_samples:
            samples = np.zeros((0, n_passes, d), dtype=np.float32)
            return means, stds, covs, samples
        return means, stds, covs

    X_t = torch.from_numpy(X).float()  # CPU tensor

    # Store outputs on CPU to avoid VRAM scaling with N
    mean_out = torch.empty(N, d, dtype=torch.float32)
    cov_out = torch.empty(N, d, d, dtype=torch.float32)

    all_samples = None
    if return_samples:
        all_samples = torch.empty(N, n_passes, d, dtype=torch.float32)

    if str(device).startswith("cuda"):
        _log_cuda_mem("before mc_predict")

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B = end - start

            xb = X_t[start:end].to(device, non_blocking=True)

            mean = torch.zeros(B, d, device=device)
            m2 = torch.zeros(B, d, d, device=device)

            for t in range(1, n_passes + 1):
                y = model(xb)  # (B, d)

                if return_samples and all_samples is not None:
                    all_samples[start:end, t - 1, :] = y.detach().cpu()

                delta = y - mean
                mean = mean + delta / t
                delta2 = y - mean
                m2 = m2 + delta.unsqueeze(-1) * delta2.unsqueeze(-2)

            cov = m2 / max(n_passes - 1, 1)

            mean_out[start:end] = mean.detach().cpu()
            cov_out[start:end] = cov.detach().cpu()

            del xb, mean, m2, cov, delta, delta2, y

    var = torch.diagonal(cov_out, dim1=-2, dim2=-1)  # (N, d)
    std_out = torch.sqrt(torch.clamp(var, min=1e-9))

    if str(device).startswith("cuda"):
        _log_cuda_mem("after mc_predict")

    if return_samples and all_samples is not None:
        return mean_out.numpy(), std_out.numpy(), cov_out.numpy(), all_samples.numpy()
    return mean_out.numpy(), std_out.numpy(), cov_out.numpy()


def predict_eval(
    model: MLP,
    X: np.ndarray,
    batch_size: int = 2048,
    device: str = "cpu",
) -> np.ndarray:
    """Single deterministic forward pass with dropout disabled."""
    model.eval()
    X = np.asarray(X)
    X_t = torch.from_numpy(X).float()
    preds = []

    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            xb = X_t[start : start + batch_size].to(device, non_blocking=True)
            preds.append(model(xb).cpu())

    model.train()
    return torch.cat(preds, dim=0).numpy()

def train_pial_multilabel_model(
    model: PiALMultilabelMLP,
    X: np.ndarray,
    docking: np.ndarray,
    selected: np.ndarray,
    remaining: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
) -> dict[str, float]:
    """Train with equally weighted MSE, selected BCE and remaining BCE."""
    dataset = PiALMultilabelDataset(X, docking, selected, remaining)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    docking_loss_fn = nn.MSELoss()
    selected_loss_fn = nn.BCEWithLogitsLoss()
    remaining_loss_fn = nn.BCEWithLogitsLoss()
    latest = {"docking": float("nan"), "selected": float("nan"), "remaining": float("nan")}

    model.train()
    for _ in range(epochs):
        sums = {name: 0.0 for name in latest}
        for xb, docking_target, selected_target, remaining_target in loader:
            xb = xb.to(device)
            docking_target = docking_target.to(device)
            selected_target = selected_target.to(device)
            remaining_target = remaining_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            docking_pred, selected_logits, remaining_logits = model(xb)
            losses = {
                "docking": docking_loss_fn(docking_pred, docking_target),
                "selected": selected_loss_fn(selected_logits, selected_target),
                "remaining": remaining_loss_fn(remaining_logits, remaining_target),
            }
            (losses["docking"] + losses["selected"] + losses["remaining"]).backward()
            optimizer.step()
            for name, loss in losses.items():
                sums[name] += float(loss.detach().cpu()) * len(xb)
        latest = {name: value / len(dataset) for name, value in sums.items()}
    latest["total"] = latest["docking"] + latest["selected"] + latest["remaining"]
    return latest


def predict_pial_outputs(
    model: PiALMultilabelMLP,
    X: np.ndarray,
    *,
    batch_size: int = 2048,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return docking, logits probabilities and expected counts deterministically."""
    model.eval()
    X_t = torch.from_numpy(np.asarray(X)).float()
    docking_parts, selected_parts, remaining_parts = [], [], []
    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            xb = X_t[start:start + batch_size].to(device, non_blocking=True)
            docking, selected_logits, remaining_logits = model(xb)
            docking_parts.append(docking.cpu())
            selected_parts.append(torch.sigmoid(selected_logits).cpu())
            remaining_parts.append(torch.sigmoid(remaining_logits).cpu())
    docking = torch.cat(docking_parts).numpy()
    selected_probs = torch.cat(selected_parts).numpy()
    remaining_probs = torch.cat(remaining_parts).numpy()
    selected_count = selected_probs.sum(axis=1)
    remaining_count = remaining_probs.sum(axis=1)
    return docking, selected_probs, remaining_probs, selected_count, remaining_count


def mc_predict_pial_counts(
    model: PiALMultilabelMLP,
    X: np.ndarray,
    *,
    n_passes: int = 50,
    batch_size: int = 2048,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply existing MC mean/covariance rules to docking and probability counts."""
    if n_passes <= 0:
        raise ValueError("n_passes must be positive")
    model.train()
    X_t = torch.from_numpy(np.asarray(X)).float()
    means, covariances = [], []
    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            xb = X_t[start:start + batch_size].to(device, non_blocking=True)
            runs = torch.stack([model.acquisition_outputs(xb) for _ in range(n_passes)])
            means.append(runs.mean(dim=0).cpu())
            centered = runs - runs.mean(dim=0, keepdim=True)
            denominator = max(n_passes - 1, 1)
            covariances.append(
                torch.einsum("tni,tnj->nij", centered, centered).div(denominator).cpu()
            )
    mean = torch.cat(means).numpy()
    covariance = torch.cat(covariances).numpy()
    std = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 1e-9))
    return mean, std, covariance
