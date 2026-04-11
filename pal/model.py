"""MLP with MC-dropout for multi-objective prediction (dimension-agnostic)."""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig

logger = logging.getLogger(__name__)


def _cuda_sync_diag_enabled() -> bool:
    """When True, call torch.cuda.synchronize() at batch boundaries for clearer GPU timings."""
    return os.environ.get("PAL_CUDA_SYNC_DIAG", "").strip() in ("1", "true", "yes")


def _predict_eval_cat_max_elems() -> int:
    """If N * d <= this, predict_eval may concat on GPU then do one D2H (see predict_eval)."""
    raw = os.environ.get("PAL_PREDICT_EVAL_CAT_MAX_ELEMS", "8000000").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 8_000_000


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


class MoleculeDataset(Dataset):
    """Simple dataset wrapping numpy feature and label arrays."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(np.asarray(X)).float()
        self.Y = torch.from_numpy(np.asarray(Y)).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


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
    num_workers: Optional[int] = None,
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

    if num_workers is None:
        nw = 8 if str(device).startswith("cuda") else 0
    else:
        nw = max(0, int(num_workers))
    pin_mem = str(device).startswith("cuda")
    dl_common = {
        "num_workers": nw,
        "pin_memory": pin_mem,
        "persistent_workers": nw > 0,
    }

    t_dl0 = time.perf_counter()
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
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_common)
    else:
        X_tr, Y_tr = X, Y
        val_loader = None
        val_ds = None

    ds = MoleculeDataset(X_tr, Y_tr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, **dl_common)
    t_dl = time.perf_counter() - t_dl0
    logger.info(
        "[TIMER] train_dataloader_setup: %.3fs n_train=%d n_val=%s pin_memory=%s num_workers=%d",
        t_dl,
        len(ds),
        len(val_ds) if val_ds is not None else 0,
        pin_mem,
        nw,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=lr_scheduler_factor, patience=lr_scheduler_patience
    )
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    wait = 0

    model.train()
    epochs_ran = 0
    sync_cuda_train = pin_mem and _cuda_sync_diag_enabled()
    for epoch in range(epochs):
        t_ep0 = time.perf_counter()
        epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        t_dl_wait = 0.0
        t_h2d = 0.0
        t_compute = 0.0
        n_tr_batches = 0
        it_tr = iter(loader)
        while True:
            t_w0 = time.perf_counter()
            try:
                xb, yb = next(it_tr)
            except StopIteration:
                break
            t_w1 = time.perf_counter()
            t_dl_wait += t_w1 - t_w0

            t_h0 = time.perf_counter()
            xb = xb.to(device, non_blocking=pin_mem)
            yb = yb.to(device, non_blocking=pin_mem)
            t_h1 = time.perf_counter()
            t_h2d += t_h1 - t_h0

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            if sync_cuda_train:
                torch.cuda.synchronize()
            t_c1 = time.perf_counter()
            t_compute += t_c1 - t_h1

            epoch_loss_sum = epoch_loss_sum + loss.detach() * len(xb)
            n_tr_batches += 1

        t_after_train = time.perf_counter()
        train_wall = t_after_train - t_ep0
        train_loss = float((epoch_loss_sum / len(ds)).item())

        if train_wall > 1e-9:
            f_dl = t_dl_wait / train_wall
            f_h2d = t_h2d / train_wall
            f_comp = t_compute / train_wall
            if f_dl > 0.22 and f_dl >= f_comp:
                tr_hint = "likely_input_pipeline_limited (dataloader wait dominates vs fwd/bwd/step wall)"
            elif f_comp > 0.32:
                tr_hint = (
                    "likely_compute_heavy_or_overlapped_gpu "
                    "(set PAL_CUDA_SYNC_DIAG=1 for stricter GPU completion timing)"
                )
            else:
                tr_hint = "mixed (overlap/H2D/loader; see fractions)"
        else:
            f_dl = f_h2d = f_comp = 0.0
            tr_hint = "n/a"

        logger.info(
            "[DIAG] train_epoch_timing epoch=%d train_wall_s=%.4f dataloader_next_s=%.4f host_to_device_s=%.4f "
            "fwd_bwd_step_s=%.4f batches=%d frac_dataloader=%.2f frac_h2d=%.2f frac_fwd_bwd_step=%.2f "
            "mean_dataloader_wait_per_batch_s=%.5f cuda_sync_diag=%s | %s",
            epoch,
            train_wall,
            t_dl_wait,
            t_h2d,
            t_compute,
            n_tr_batches,
            f_dl,
            f_h2d,
            f_comp,
            t_dl_wait / max(n_tr_batches, 1),
            sync_cuda_train,
            tr_hint,
        )

        if use_val and val_loader is not None and val_ds is not None:
            model.eval()
            val_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
            val_preds = []
            val_trues = []
            t_v_dl = 0.0
            t_v_h2d = 0.0
            t_v_comp = 0.0
            n_va_batches = 0
            with torch.inference_mode():
                it_va = iter(val_loader)
                while True:
                    t_w0 = time.perf_counter()
                    try:
                        xb, yb = next(it_va)
                    except StopIteration:
                        break
                    t_w1 = time.perf_counter()
                    t_v_dl += t_w1 - t_w0
                    t_h0 = time.perf_counter()
                    xb = xb.to(device, non_blocking=pin_mem)
                    yb = yb.to(device, non_blocking=pin_mem)
                    t_h1 = time.perf_counter()
                    t_v_h2d += t_h1 - t_h0
                    pred = model(xb)
                    val_loss_sum = val_loss_sum + criterion(pred, yb) * len(xb)
                    val_preds.append(pred)
                    val_trues.append(yb)
                    if sync_cuda_train:
                        torch.cuda.synchronize()
                    t_c1 = time.perf_counter()
                    t_v_comp += t_c1 - t_h1
                    n_va_batches += 1

            val_loss = float((val_loss_sum / len(val_ds)).item())
            val_wall = sum((t_v_dl, t_v_h2d, t_v_comp))
            if val_wall > 1e-9:
                logger.info(
                    "[DIAG] val_epoch_timing epoch=%d val_wall_s=%.4f dataloader_next_s=%.4f host_to_device_s=%.4f "
                    "forward_s=%.4f batches=%d frac_dataloader=%.2f frac_h2d=%.2f frac_forward=%.2f",
                    epoch,
                    val_wall,
                    t_v_dl,
                    t_v_h2d,
                    t_v_comp,
                    n_va_batches,
                    t_v_dl / val_wall,
                    t_v_h2d / val_wall,
                    t_v_comp / val_wall,
                )

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

        epochs_ran = epoch + 1
        dt_ep = time.perf_counter() - t_ep0
        if use_val and val_loader is not None and val_ds is not None:
            logger.info(
                "[TIMER] train_epoch: epoch=%d dt=%.3fs train_loss=%.6g val_loss=%.6g",
                epoch,
                dt_ep,
                train_loss,
                monitored,
            )
        else:
            logger.info(
                "[TIMER] train_epoch: epoch=%d dt=%.3fs train_loss=%.6g",
                epoch,
                dt_ep,
                train_loss,
            )

        if epoch >= min_epochs and patience > 0 and wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info(
        "Train done: n=%d use_val=%s best_loss=%.6g epochs_ran=%d device=%s out_features=%d",
        n_samples,
        use_val,
        best_loss,
        epochs_ran,
        device,
        model.out_features,
    )
    return best_loss


def mc_predict(
    model: MLP,
    X: np.ndarray,
    n_passes: int = 50,
    batch_size: int = 4096,
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

    nb = str(device).startswith("cuda")
    sync_b = nb and _cuda_sync_diag_enabled()
    prep_ts: list[float] = []
    fwd_ts: list[float] = []
    d2h_ts: list[float] = []
    t_mc0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B = end - start

            t_p0 = time.perf_counter()
            xb = X_t[start:end].to(device, non_blocking=nb)
            if sync_b:
                torch.cuda.synchronize()
            t_p1 = time.perf_counter()
            prep_ts.append(t_p1 - t_p0)

            t_f0 = time.perf_counter()
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
            if sync_b:
                torch.cuda.synchronize()
            t_f1 = time.perf_counter()
            fwd_ts.append(t_f1 - t_f0)

            t_d0 = time.perf_counter()
            mean_out[start:end] = mean.detach().cpu()
            cov_out[start:end] = cov.detach().cpu()
            if sync_b:
                torch.cuda.synchronize()
            t_d1 = time.perf_counter()
            d2h_ts.append(t_d1 - t_d0)

            del xb, mean, m2, cov, delta, delta2, y

    var = torch.diagonal(cov_out, dim1=-2, dim2=-1)  # (N, d)
    std_out = torch.sqrt(torch.clamp(var, min=1e-9))
    t_mc = time.perf_counter() - t_mc0
    n_batch = (N + batch_size - 1) // batch_size
    logger.info(
        "[TIMER] mc_predict: %.3fs N=%d batch_size=%d n_passes=%d device=%s batches=%d forwards≈%d",
        t_mc,
        N,
        batch_size,
        n_passes,
        device,
        n_batch,
        n_batch * n_passes,
    )
    if prep_ts:
        mp = sum(prep_ts) / len(prep_ts)
        mf = sum(fwd_ts) / len(fwd_ts)
        md = sum(d2h_ts) / len(d2h_ts)
        tot_b = mp + mf + md
        host_pipe = mp + md
        if sync_b and tot_b > 1e-9:
            if host_pipe >= 0.38 * tot_b and host_pipe >= mf:
                mc_hint = "likely_cpu_or_transfer_bound (mean prep+d2h vs MC-forward block)"
            elif mf >= 0.55 * tot_b:
                mc_hint = "likely_gpu_forward_bound (MC passes dominate batch wall)"
            else:
                mc_hint = "mixed_pipeline"
        else:
            mc_hint = (
                "overlap_unknown: set PAL_CUDA_SYNC_DIAG=1 on CUDA to separate H2D/GPU/D2H "
                "(wall clocks overlap when unsynchronized)"
            )
        logger.info(
            "[DIAG] mc_predict_batch cuda_sync=%s batches=%d mean_batch_prep_s=%.6f mean_mc_gpu_block_s=%.6f "
            "mean_d2h_s=%.6f mean_per_batch_wall_s=%.6f est_host_between_batches_s≈mean_d2h+mean_prep=%.6f | %s",
            sync_b,
            len(prep_ts),
            mp,
            mf,
            md,
            tot_b,
            md + mp,
            mc_hint,
        )

    if str(device).startswith("cuda"):
        _log_cuda_mem("after mc_predict")

    if return_samples and all_samples is not None:
        return mean_out.numpy(), std_out.numpy(), cov_out.numpy(), all_samples.numpy()
    return mean_out.numpy(), std_out.numpy(), cov_out.numpy()


def predict_eval(
    model: MLP,
    X: np.ndarray,
    batch_size: int = 4096,
    device: str = "cpu",
) -> np.ndarray:
    """Single deterministic forward pass with dropout disabled.

    On CUDA, when ``N * out_features`` is small enough (see ``PAL_PREDICT_EVAL_CAT_MAX_ELEMS``),
    activations are concatenated on GPU and transferred to CPU once to avoid per-batch D2H.
    """
    model.eval()
    X = np.asarray(X)
    N = len(X)
    d_out = int(model.out_features)
    if N == 0:
        model.train()
        return np.zeros((0, d_out), dtype=np.float32)

    X_t = torch.from_numpy(X).float()

    nb = str(device).startswith("cuda")
    sync_b = nb and _cuda_sync_diag_enabled()
    max_cat_elems = _predict_eval_cat_max_elems()
    use_gpu_cat = bool(nb and max_cat_elems > 0 and N * d_out <= max_cat_elems)

    gpu_acc: list[torch.Tensor] = []
    preds_cpu: list[torch.Tensor] = []
    prep_ts: list[float] = []
    fwd_ts: list[float] = []
    d2h_ts: list[float] = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(X_t), batch_size):
            t_p0 = time.perf_counter()
            xb = X_t[start : start + batch_size].to(device, non_blocking=nb)
            if sync_b:
                torch.cuda.synchronize()
            t_p1 = time.perf_counter()
            prep_ts.append(t_p1 - t_p0)

            t_f0 = time.perf_counter()
            y = model(xb)
            if sync_b:
                torch.cuda.synchronize()
            t_f1 = time.perf_counter()
            fwd_ts.append(t_f1 - t_f0)

            if use_gpu_cat:
                gpu_acc.append(y)
            else:
                t_d0 = time.perf_counter()
                preds_cpu.append(y.cpu())
                if sync_b:
                    torch.cuda.synchronize()
                t_d1 = time.perf_counter()
                d2h_ts.append(t_d1 - t_d0)

    if use_gpu_cat:
        t_cat0 = time.perf_counter()
        out_np = torch.cat(gpu_acc, dim=0).cpu().numpy()
        if sync_b:
            torch.cuda.synchronize()
        total_d2h = time.perf_counter() - t_cat0
        logger.info(
            "[DIAG] predict_eval: gpu_cat_single_d2h N*d=%d max_elems=%d cat_plus_d2h_s=%.6f batches=%d",
            N * d_out,
            max_cat_elems,
            total_d2h,
            len(prep_ts),
        )

    model.train()
    dt = time.perf_counter() - t0
    n_b = len(prep_ts)
    logger.info(
        "[TIMER] predict_eval: %.3fs N=%d batch_size=%d device=%s gpu_cat=%s",
        dt,
        N,
        batch_size,
        device,
        use_gpu_cat,
    )
    if prep_ts and not use_gpu_cat:
        mp = sum(prep_ts) / n_b
        mf = sum(fwd_ts) / n_b
        md = sum(d2h_ts) / n_b
        tot_b = mp + mf + md
        host_pipe = mp + md
        if sync_b and tot_b > 1e-9:
            if host_pipe >= 0.38 * tot_b and host_pipe >= mf:
                pe_hint = "likely_cpu_or_transfer_bound"
            elif mf >= 0.55 * tot_b:
                pe_hint = "likely_gpu_forward_bound"
            else:
                pe_hint = "mixed_pipeline"
        else:
            pe_hint = (
                "overlap_unknown: set PAL_CUDA_SYNC_DIAG=1 on CUDA for H2D/GPU/D2H split "
                "(wall clocks overlap when unsynchronized)"
            )
        logger.info(
            "[DIAG] predict_eval_batch cuda_sync=%s batches=%d mean_batch_prep_s=%.6f mean_forward_s=%.6f "
            "mean_d2h_s=%.6f mean_per_batch_wall_s=%.6f est_idle_between_batches_s≈mean_d2h+mean_prep=%.6f | %s",
            sync_b,
            n_b,
            mp,
            mf,
            md,
            tot_b,
            md + mp,
            pe_hint,
        )
    elif prep_ts and use_gpu_cat:
        mp = sum(prep_ts) / n_b
        mf = sum(fwd_ts) / n_b
        logger.info(
            "[DIAG] predict_eval_batch cuda_sync=%s batches=%d mean_batch_prep_s=%.6f mean_forward_s=%.6f "
            "d2h_mode=single_concat (see gpu_cat_single_d2h) | %s",
            sync_b,
            n_b,
            mp,
            mf,
            "gpu_cat_path",
        )

    if use_gpu_cat:
        return out_np
    return torch.cat(preds_cpu, dim=0).numpy()