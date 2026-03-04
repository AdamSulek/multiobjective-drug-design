"""Core active-learning loop."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from .acquisition.base import AcquisitionFunction
from .config import ExperimentConfig
from .model import build_model, mc_predict, predict_eval, train_model
from .pareto import batch_delta_hv_3d, pareto_front
from .pareto_3D import (
    pareto_front_3d_max,
    hv_3d_max,
    build_dominance_index_3d,
)

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@contextmanager
def timed(stage: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    logging.info(f"[TIMER] {stage}: {dt:.3f}s")


def _hv(Y: np.ndarray, ref_point: tuple[float, ...]) -> float:
    m = Y.shape[1]
    if m == 2:
        return hypervolume_2d(Y, ref_point)
    if m == 3:
        front = pareto_front_3d_max(Y)
        return hv_3d_max(front, ref_point)
    raise ValueError(f"Only 2D/3D supported, got m={m}")


def _nan_metrics(n_obj: int) -> dict:
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
    Y_labeled: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
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
    acq_fn: AcquisitionFunction,
    config: ExperimentConfig,
    seed: int,
    seed_indices: np.ndarray | None = None,
):
    """Run a full active-learning loop."""
    N = len(Y_pool)
    n_obj = int(Y_pool.shape[1])
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
    hv0 = _hv(state.Y_labeled, al.ref_point)
    state.hv_history.append(hv0)
    state.acq_time_per_iter.append(0.0)  # no acquisition at seed
    state.train_metrics.append(_nan_metrics(n_obj))
    state.val_metrics.append(_nan_metrics(n_obj))
    state.sel_metrics.append(_nan_metrics(n_obj))

    logging.info(
        f"[{acq_fn.name}] seed  | "
        f"labeled={len(state.labeled_indices):4d}  HV={hv0:.4f}  acq_time=0.000s"
    )

    # --- main loop ---
    for it in range(1, al.n_iterations + 1):
        state.iteration = it

        # 1) train model from scratch on labeled set (normalize targets)
        with timed("build_model"):
            model = build_model(config.model, device=config.device, out_features=Y_pool.shape[1])

        X_train = X_pool[state.labeled_indices]
        Y_train = state.Y_labeled

        Y_mean = Y_train.mean(axis=0)
        Y_std = np.maximum(Y_train.std(axis=0), 1e-8)
        Y_train_norm = (Y_train - Y_mean) / Y_std

        with timed("train_model"):
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
        with timed("predict_eval(train)"):
            Y_train_pred = predict_eval(model, X_train, device=config.device)
        Y_train_pred = Y_train_pred * Y_std + Y_mean
        train_m = compute_regression_metrics(Y_train, Y_train_pred)
        state.train_metrics.append(train_m)

        # 2) Predictions on unlabeled pool
        X_unlabeled = X_pool[state.unlabeled_indices]

        #is_fast_ellipse = ("FastEllipse3D" in acq_fn.name) or ("ellipse_fast" in acq_fn.name.lower())
        needs_cov = getattr(acq_fn, "needs_full_cov", False)

        # For huge pools:
        # - if strategy needs full covariance (ellipse*), do cheap mean on whole pool,
        #   and MC+cov only on top-K later.
        # - otherwise do normal MC on whole pool (what you had).
        if needs_cov:
            # --- A) cheap global pass (NO MC, NO cov) ---
            with timed("predict_eval(unlabeled)"):
                means = predict_eval(model, X_unlabeled, device=config.device)   # (U,d)
            means = means * Y_std + Y_mean
            stds = None
            covs = None
        else:
            # --- normal path (MC on whole unlabeled) ---
            with timed("mc_predict(unlabeled)"):
                means, stds, covs = mc_predict(
                    model,
                    X_unlabeled,
                    n_passes=mcfg.mc_passes,
                    device=config.device,
                )
            means = means * Y_std + Y_mean
            stds = stds * Y_std
            covs = covs * np.outer(Y_std, Y_std)[None, :, :]

        # --- Pareto/Fenwick precompute for this iteration (3D only) ---
        pareto_front = None
        hv_front = None
        dom_index = None

        if n_obj == 3 and state.Y_labeled.shape[0] > 0:
            with timed("pareto_front_3d_max(labeled)"):
                pareto_front = pareto_front_3d_max(state.Y_labeled)

            with timed("hv_3d_max(front)"):
                hv_front = hv_3d_max(pareto_front, al.ref_point)

            with timed("build_dominance_index_3d(front)"):
                dom_index = build_dominance_index_3d(pareto_front)

            logging.info(
                f"[PARETO] it={it} labeled={state.Y_labeled.shape[0]} "
                f"front={pareto_front.shape[0]} hv_front={hv_front:.6f}"
            )

        # 3) acquisition: select next batch (indices local to unlabeled list)
        # --- Special 2-stage path for FastEllipse: cov only on top-K ---
        top_local = None
        means_top = None
        stds_top = None
        covs_top = None

        if needs_cov:
            U = len(state.unlabeled_indices)

            # 1) build current front once
            front_now = pareto_front_3d_max(state.Y_labeled.astype(np.float32))

            # 2) cheap score on means (delta-HV on mean)
            logging.info(f"[TOPK] starting cheap_score: U={U}")

            with timed("cheap_score(deltaHV on mean)"):
                ref = np.asarray(al.ref_point, dtype=np.float32)
                cheap = np.min(means.astype(np.float32) - ref[None, :], axis=1)

            logging.info(f"[TOPK] cheap_score done. cheap shape={cheap.shape} max={float(np.max(cheap)):.6f}")

            # 3) choose top-K candidates (mix top + random tail)
            K = int(getattr(al, "ellipse_topk", 2000))  # or hardcode 2000 if you prefer
            K = min(K, U)

            frac_top = 0.8
            K_top = int(frac_top * K)
            K_rand = K - K_top

            top_part = np.argpartition(cheap, -K_top)[-K_top:]

            if K_rand > 0:
                rng2 = np.random.default_rng(seed + 10_000 + it)
                mask = np.ones(U, dtype=bool)
                mask[top_part] = False
                rest = np.flatnonzero(mask)
                if rest.size > 0:
                    rand_part = rng2.choice(rest, size=min(K_rand, rest.size), replace=False)
                    top_local = np.concatenate([top_part, rand_part])
                else:
                    top_local = top_part
            else:
                top_local = top_part

            # 4) MC+cov ONLY on top_local
            X_top = X_unlabeled[top_local]
            logging.info(f"[TOPK] it={it} U={U} K={len(top_local)} (top={K_top}, rand={len(top_local)-K_top})")
            
            with timed("mc_predict(topK cov)"):
                means_top, stds_top, covs_top = mc_predict(
                    model,
                    X_top,
                    n_passes=mcfg.mc_passes,
                    device=config.device,
                )

            means_top = means_top * Y_std + Y_mean
            stds_top = stds_top * Y_std
            covs_top = covs_top * np.outer(Y_std, Y_std)[None, :, :]
    
        t0 = time.perf_counter()
        with timed("acquisition.select()"):
            if needs_cov:
                logging.info(
                    f"[ACQ] needs_cov=True: calling select on TOPK "
                    f"means_top={means_top.shape} stds_top={stds_top.shape} covs_top={covs_top.shape}"
                )
                # select within topK space
                sel_local_top = acq_fn.select(
                    means_top,
                    stds_top,
                    state.Y_labeled,
                    al.ref_point,
                    k=al.batch_size,
                    covs=covs_top,
                    pareto_front=pareto_front,
                    pareto_dom_index=dom_index,
                    pareto_hv=hv_front,
                )
                sel_local_top = np.asarray(sel_local_top).astype(int).ravel()
                sel_local = top_local[sel_local_top]   # map back to unlabeled-local indices
            else:
                cov_shape = None if covs is None else covs.shape
                std_shape = None if stds is None else stds.shape
                logging.info(
                    f"[ACQ] needs_cov=False: calling select on FULL "
                    f"means={means.shape} stds={std_shape} covs={cov_shape}"
                )
                sel_local = acq_fn.select(
                    means,
                    stds,
                    state.Y_labeled,
                    al.ref_point,
                    k=al.batch_size,
                    covs=covs,
                    pareto_front=pareto_front,
                    pareto_dom_index=dom_index,
                    pareto_hv=hv_front,
                )
        state.acq_time_per_iter.append(time.perf_counter() - t0)

        # --- normalize sel_local to 1D int array
        sel_local = np.asarray(sel_local).astype(int).ravel()
        U = len(state.unlabeled_indices)
        N = len(Y_pool)

        if sel_local.size == 0:
            raise RuntimeError("acq_fn.select returned empty selection")

        # If indices are out of range for unlabeled, assume they are POOL indices and map -> local
        if sel_local.min() < 0 or sel_local.max() >= U:
            if sel_local.min() >= 0 and sel_local.max() < N:
                pos = {pool_idx: j for j, pool_idx in enumerate(state.unlabeled_indices)}
                try:
                    sel_local = np.array([pos[p] for p in sel_local], dtype=int)
                except KeyError as e:
                    raise RuntimeError(
                        f"acq_fn.select returned pool index not in unlabeled set: {e}. "
                        "Strategy is selecting already-labeled points."
                    ) from e
            else:
                raise RuntimeError(
                    f"sel_local indices out of range and not valid pool indices: "
                    f"min={sel_local.min()} max={sel_local.max()} "
                    f"(unlabeled size U={U}, pool size N={N})"
                )

        # Optionally enforce unique + correct count
        sel_local = np.unique(sel_local)
        if sel_local.size > al.batch_size:
            sel_local = sel_local[-al.batch_size:]

        # map local -> pool
        sel_pool = [state.unlabeled_indices[i] for i in sel_local]

        state.selections_per_iter.append(list(sel_pool))
        # 4) "label" — oracle lookup
        new_labels = Y_pool[sel_pool]

        # selected-compounds metrics (MC means vs GT for selected points)
        if needs_cov:
            # sel_local are indices in unlabeled space; map them into top_local positions
            pos = {int(u): j for j, u in enumerate(top_local)}
            sel_pos = np.array([pos[int(u)] for u in sel_local], dtype=int)
            sel_pred = means_top[sel_pos]
        else:
            sel_pred = means[sel_local]

        sel_m = compute_regression_metrics(new_labels, sel_pred)
        
        state.sel_metrics.append(sel_m)

        # snapshot of current unlabeled pool for validation metrics
        unlabeled_snapshot = list(state.unlabeled_indices)

        # 5) update labeled/unlabeled sets
        sel_pool_set = set(sel_pool)

        state.labeled_indices.extend(sel_pool)
        state.Y_labeled = np.vstack([state.Y_labeled, new_labels])

        state.unlabeled_indices = [i for i in state.unlabeled_indices if i not in sel_pool_set]

        # 6) validation metrics every val_every
        if it % al.val_every == 0:
            Y_unlabeled_true = Y_pool[unlabeled_snapshot]
            val_m = compute_regression_metrics(Y_unlabeled_true, means)
        else:
            val_m = _nan_metrics(n_obj)
        state.val_metrics.append(val_m)

        # 7) hypervolume (overall labeled)
        with timed("hv(labeled_total)"):
            hv = _hv(state.Y_labeled, al.ref_point)
            hv_gain = hv - state.hv_history[-1]
        state.hv_history.append(hv)

        # console output
        def _fmt_list(xs, fmt):
            return "[" + ",".join(format(float(x), fmt) for x in xs) + "]"

        tmse = _fmt_list(train_m["mse"], ".4f")
        tr2  = _fmt_list(train_m["r2"],  ".2f")

        line = (
            f"[{acq_fn.name}] it {it:3d} | "
            f"labeled={len(state.labeled_indices):4d}  HV={hv:.4f}  (+{hv_gain:.4f})"
            f"acq={state.acq_time_per_iter[-1]:.3f}s"
            f"  train_MSE={tmse} R2={tr2}"
        )

        if not np.isnan(val_m["mse"][0]):
            vmse = _fmt_list(val_m["mse"], ".4f")
            vr2  = _fmt_list(val_m["r2"],  ".2f")
            line += f"  val_MSE={vmse} R2={vr2}"
        logging.info(line)

    return state