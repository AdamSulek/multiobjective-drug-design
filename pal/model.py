"""MLP with MC-dropout for multi-objective prediction."""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig


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
        self.head = nn.Linear(prev, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class MoleculeDataset(Dataset):
    """Simple dataset wrapping numpy feature and label arrays."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def build_model(cfg: ModelConfig, device: str = "cpu") -> MLP:
    """Construct an MLP from config and move to device."""
    return MLP(
        in_features=cfg.in_features,
        hidden_sizes=cfg.hidden_sizes,
        dropout=cfg.dropout,
        out_features=cfg.out_features,
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
    # re-initialise weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    n_samples = len(X)

    # --- internal train/val split ---
    use_val = n_samples >= 10 and patience > 0
    if use_val:
        rng = np.random.RandomState(n_samples)  # deterministic per size
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

    ds = MoleculeDataset(X_tr, Y_tr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=lr_scheduler_factor, patience=lr_scheduler_patience,
    )
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    wait = 0

    model.train()
    for epoch in range(epochs):
        # --- training ---
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_loss = epoch_loss / len(ds)

        # --- monitored loss ---
        if use_val:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(model(xb), yb).item() * len(xb)
            val_loss /= len(val_ds)
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

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_loss


def mc_predict(
    model: MLP,
    X: np.ndarray,
    n_passes: int = 50,
    batch_size: int = 2048,
    device: str = "cpu",
    return_samples: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run MC-dropout inference using Welford's online algorithm.

    Parameters
    ----------
    model : MLP
        Trained model (dropout layers will be kept active).
    X : np.ndarray
        Input features, shape ``(N, D)``.
    n_passes : int
        Number of stochastic forward passes.
    batch_size : int
        Inference batch size.
    device : str
        Torch device.
    return_samples : bool
        If True, also return the raw predictions from every MC pass
        as an ``(N, n_passes, 2)`` array appended to the return tuple.

    Returns
    -------
    means : np.ndarray, shape ``(N, 2)``
    stds  : np.ndarray, shape ``(N, 2)``
    covs  : np.ndarray, shape ``(N, 2, 2)``
    samples : np.ndarray, shape ``(N, n_passes, 2)``  *(only when return_samples=True)*
    """
    model.train()  # keep dropout active

    N = len(X)
    X_t = torch.from_numpy(X).float()

    # Welford accumulators (full covariance via outer product)
    mean = torch.zeros(N, 2)
    m2 = torch.zeros(N, 2, 2)

    if return_samples:
        all_samples = torch.zeros(N, n_passes, 2)

    with torch.no_grad():
        for t in range(1, n_passes + 1):
            preds = []
            for start in range(0, N, batch_size):
                xb = X_t[start : start + batch_size].to(device)
                preds.append(model(xb).cpu())
            y = torch.cat(preds, dim=0)  # (N, 2)

            if return_samples:
                all_samples[:, t - 1, :] = y

            delta = y - mean
            mean = mean + delta / t
            delta2 = y - mean
            m2 = m2 + delta.unsqueeze(-1) * delta2.unsqueeze(-2)  # (N, 2, 2)

    cov = m2 / max(n_passes - 1, 1)
    var = torch.diagonal(cov, dim1=-2, dim2=-1)  # (N, 2)
    std = torch.sqrt(torch.clamp(var, min=1e-9))

    if return_samples:
        return mean.numpy(), std.numpy(), cov.numpy(), all_samples.numpy()
    return mean.numpy(), std.numpy(), cov.numpy()


def predict_eval(
    model: MLP,
    X: np.ndarray,
    batch_size: int = 2048,
    device: str = "cpu",
) -> np.ndarray:
    """Single deterministic forward pass with dropout disabled.

    Parameters
    ----------
    model : MLP
        Trained model.
    X : np.ndarray
        Input features, shape ``(N, D)``.
    batch_size : int
        Inference batch size.
    device : str
        Torch device.

    Returns
    -------
    np.ndarray, shape ``(N, 2)``
    """
    model.eval()
    X_t = torch.from_numpy(X).float()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            xb = X_t[start : start + batch_size].to(device)
            preds.append(model(xb).cpu())
    model.train()
    return torch.cat(preds, dim=0).numpy()
