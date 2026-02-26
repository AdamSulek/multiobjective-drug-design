"""Core active-learning loop."""

import time
from dataclasses import dataclass, field

import numpy as np

from .acquisition.base import AcquisitionFunction
from .config import ExperimentConfig
from .model import build_model, mc_predict, predict_eval, train_model
from .pareto import hypervolume_2d


def _nan_metrics(n_obj: int = 2) -> dict:
    """Return a metrics dict filled with NaN (for iterations with no model)."""
    nan_list = [float("nan")] * n_obj
    return {"mse": list(nan_list), "mae": list(nan_list), "r2": list(nan_list)}


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MSE, MAE, R2 per objective column.

    Parameters
    ----------
    y_true : np.ndarray, shape ``(N, n_obj)``
    y_pred : np.ndarray, shape ``(N, n_obj)``

    Returns
    -------
    dict with keys ``"mse"``, ``"mae"``, ``"r2"``; each a list of floats.
    """
    residuals = y_true - y_pred
    mse = (residuals ** 2).mean(axis=0).tolist()
    mae = np.abs(residuals).mean(axis=0).tolist()
    ss_res = (residuals ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = (1.0 - ss_res / np.maximum(ss_tot, 1e-12)).tolist()
    return {"mse": mse, "mae": mae, "r2": r2}


@dataclass
class ALState:
    """Mutable state of one AL run."""

    labeled_indices: list[int] = field(default_factory=list)
    unlabeled_indices: list[int] = field(default_factory=list)
    Y_labeled: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    hv_history: list[float] = field(default_factory=list)
    selections_per_iter: list[list[int]] = field(default_factory=list)
    acq_time_per_iter: list[float] = field(default_factory=list)
    train_metrics: list[dict] = field(default_factory=list)
    val_metrics: list[dict] = field(default_factory=list)
    sel_metrics: list[dict] = field(default_factory=list)
    iteration: int = 0


def run_al_loop(
    X_pool,
    Y_pool,
    acq_fn,
    config,
    seed: int,
    seed_indices: np.ndarray | None = None,
):
    """Run a full active-learning loop.

    Parameters
    ----------
    X_pool : np.ndarray or LazyECFP-like, shape (N, D)
        Features for the entire pool.
    Y_pool : np.ndarray, shape (N, 2)
        Ground-truth objective values (oracle look-up table).
    acq_fn : AcquisitionFunction
        Strategy used to select the next batch.
    config : ExperimentConfig
        Full experiment configuration.
    seed : int
        Random seed for the initial labeled set (and any internal randomness if used).
    seed_indices : np.ndarray | None
        Optional explicit initial labeled indices (length = seed_size).

    Returns
    -------
    ALState
        Final state including HV history.
    """
    import time  # upewnij się, że masz; możesz też przenieść do importów na górze pliku

    N = len(Y_pool)
    al = config.al
    mcfg = config.model

    rng = np.random.default_rng(seed)

    # --- choose initial labeled set (seed) ---
    if seed_indices is None:
        labeled = rng.choice(N, size=al.seed_size, replace=False).astype(int).tolist()
    else:
        labeled_arr = np.array(seed_indices, dtype=int).ravel()

        if len(labeled_arr) != al.seed_size:
            raise ValueError(
                f"seed_indices length={len(labeled_arr)} but seed_size={al.seed_size}"
            )
        if labeled_arr.min() < 0 or labeled_arr.max() >= N:
            raise ValueError(f"seed_indices out of range [0, {N-1}]")
        if len(np.unique(labeled_arr)) != len(labeled_arr):
            raise ValueError("seed_indices contain duplicates")

        labeled = labeled_arr.tolist()

    labeled_set = set(labeled)
    unlabeled = [i for i in range(N) if i not in labeled_set]

    # --- init state ---
    state = ALState(
        labeled_indices=list(labeled),
        unlabeled_indices=list(unlabeled),
        Y_labeled=Y_pool[labeled].copy(),
    )

    # record seed as iteration 0
    state.selections_per_iter.append(list(labeled))

    # initial HV
    hv0 = hypervolume_2d(state.Y_labeled, al.ref_point)
    state.hv_history.append(hv0)
    state.acq_time_per_iter.append(0.0)  # no acquisition at seed
    state.train_metrics.append(_nan_metrics())
    state.val_metrics.append(_nan_metrics())
    state.sel_metrics.append(_nan_metrics())

    print(
        f"[{acq_fn.name}] seed  | "
        f"labeled={len(state.labeled_indices):4d}  HV={hv0:.4f}  acq_time=0.000s"
    )

    # --- main loop ---
    for it in range(1, al.n_iterations + 1):
        state.iteration = it

        # 1) train model from scratch on labeled set (normalize targets)
        model = build_model(mcfg, device=config.device)

        X_train = X_pool[state.labeled_indices]
        Y_train = state.Y_labeled

        Y_mean = Y_train.mean(axis=0)
        Y_std = np.maximum(Y_train.std(axis=0), 1e-8)
        Y_train_norm = (Y_train - Y_mean) / Y_std

        train_model(
            model,
            X_train,
            Y_train_norm,
            epochs=mcfg.epochs,
            batch_size=mcfg.batch_size,
            lr=mcfg.lr,
            weight_decay=mcfg.weight_decay,
            device=config.device,
            patience=mcfg.patience,
            min_epochs=mcfg.min_epochs,
            val_fraction=mcfg.val_fraction,
            lr_scheduler_patience=mcfg.lr_scheduler_patience,
            lr_scheduler_factor=mcfg.lr_scheduler_factor,
        )

        # training metrics (denormalize predictions)
        Y_train_pred = predict_eval(model, X_train, device=config.device)
        Y_train_pred = Y_train_pred * Y_std + Y_mean
        train_m = compute_regression_metrics(Y_train, Y_train_pred)
        state.train_metrics.append(train_m)

        # 2) MC-dropout predictions on unlabeled pool (denormalize)
        X_unlabeled = X_pool[state.unlabeled_indices]
        means, stds, covs = mc_predict(
            model,
            X_unlabeled,
            n_passes=mcfg.mc_passes,
            device=config.device,
        )
        means = means * Y_std + Y_mean
        stds = stds * Y_std
        covs = covs * np.outer(Y_std, Y_std)[None, :, :]

        # 3) acquisition: select next batch (indices local to unlabeled list)
        t0 = time.perf_counter()
        sel_local = acq_fn.select(
            means, stds, state.Y_labeled, al.ref_point, k=al.batch_size, covs=covs
        )
        acq_elapsed = time.perf_counter() - t0
        state.acq_time_per_iter.append(acq_elapsed)

        # map local -> pool indices
        sel_pool = [state.unlabeled_indices[i] for i in sel_local]
        state.selections_per_iter.append(list(sel_pool))

        # 4) "label" — oracle lookup
        new_labels = Y_pool[sel_pool]

        # selected-compounds metrics (MC means vs GT for selected points)
        sel_m = compute_regression_metrics(new_labels, means[sel_local])
        state.sel_metrics.append(sel_m)

        # snapshot of current unlabeled pool for validation metrics
        unlabeled_snapshot = list(state.unlabeled_indices)

        # 5) update labeled/unlabeled sets
        sel_pool_set = set(sel_pool)

        state.labeled_indices.extend(sel_pool)
        state.Y_labeled = np.vstack([state.Y_labeled, new_labels])

        # keep order, remove selected efficiently
        state.unlabeled_indices = [i for i in state.unlabeled_indices if i not in sel_pool_set]

        # 6) validation metrics (on full unlabeled snapshot) every val_every
        if it % al.val_every == 0:
            Y_unlabeled_true = Y_pool[unlabeled_snapshot]
            val_m = compute_regression_metrics(Y_unlabeled_true, means)
        else:
            val_m = _nan_metrics()
        state.val_metrics.append(val_m)

        # 7) hypervolume
        hv = hypervolume_2d(state.Y_labeled, al.ref_point)
        state.hv_history.append(hv)

        # console output
        tmse = f"[{train_m['mse'][0]:.4f},{train_m['mse'][1]:.4f}]"
        tr2 = f"[{train_m['r2'][0]:.2f},{train_m['r2'][1]:.2f}]"
        line = (
            f"[{acq_fn.name}] it {it:3d} | "
            f"labeled={len(state.labeled_indices):4d}  HV={hv:.4f}  acq={acq_elapsed:.3f}s"
            f"  train_MSE={tmse} R2={tr2}"
        )
        if not np.isnan(val_m["mse"][0]):
            vmse = f"[{val_m['mse'][0]:.4f},{val_m['mse'][1]:.4f}]"
            vr2 = f"[{val_m['r2'][0]:.2f},{val_m['r2'][1]:.2f}]"
            line += f"  val_MSE={vmse} R2={vr2}"
        print(line)

    return state
