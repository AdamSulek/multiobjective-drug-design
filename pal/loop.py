"""Core active-learning loop."""

import inspect
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from .acquisition.base import AcquisitionFunction
from .config import ExperimentConfig
from .model import build_model, mc_predict, predict_eval, train_model
from .pareto import hypervolume_2d, hypervolume_3d_max_fast, pareto_front_max_3d_fast


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _get_logger() -> logging.Logger:
    pal_logger = logging.getLogger("pal")
    if pal_logger.handlers:
        return pal_logger
    return logging.getLogger()


LOGGER = _get_logger()


@contextmanager
def timed(stage: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    LOGGER.info(f"[TIMER] {stage}: {dt:.3f}s")


def _hv(Y: np.ndarray, ref_point: tuple[float, ...]) -> float:
    m = Y.shape[1]
    if m == 2:
        return hypervolume_2d(Y, ref_point)
    if m == 3:
        return hypervolume_3d_max_fast(Y, ref_point)
    raise ValueError(f"Only 2D/3D supported, got m={m}")


def _nan_metrics(n_obj: int) -> dict:
    nan_list = [float("nan")] * n_obj
    return {"mse": list(nan_list), "mae": list(nan_list), "r2": list(nan_list)}


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals = y_true - y_pred
    mse = (residuals ** 2).mean(axis=0).tolist()
    mae = np.abs(residuals).mean(axis=0).tolist()
    ss_res = (residuals ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = (1.0 - ss_res / np.maximum(ss_tot, 1e-12)).tolist()
    return {"mse": mse, "mae": mae, "r2": r2}


def _summary_line(vals: list[float]) -> str:
    if not vals:
        return "sum=0.000s mean=0.000s std=0.000s min=0.000s max=0.000s"
    arr = np.asarray(vals, dtype=float)
    return (
        f"sum={arr.sum():.3f}s mean={arr.mean():.3f}s std={arr.std(ddof=0):.3f}s "
        f"min={arr.min():.3f}s max={arr.max():.3f}s"
    )


STAGE_KEYS = (
    "total",
    "build",
    "train",
    "pred_train",
    "pred_unlab",
    "cov_reconstruct",
    "acq",
    "hv",
)


def _log_timing_summary(acq_name: str, state: "ALState") -> None:
    if not state.iter_stage_times:
        return

    LOGGER.info(
        f"[TIME_SUMMARY] [{acq_name}] n_iters={len(state.iter_stage_times)} "
        f"n_labeled_final={len(state.labeled_indices)}"
    )
    for key in STAGE_KEYS:
        vals = [float(x.get(key, 0.0)) for x in state.iter_stage_times]
        LOGGER.info(f"[TIME_SUMMARY] [{acq_name}] stage={key} {_summary_line(vals)}")


def _predict_unlabeled(
    *,
    model,
    X_unlabeled,
    mcfg,
    device: str,
    needs_cov: bool,
    needs_uncertainty: bool,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    if needs_cov or needs_uncertainty:
        stage = "mc_predict(unlabeled_full_with_cov)" if needs_cov else "mc_predict(unlabeled)"
        t0 = time.perf_counter()
        with timed(stage):
            means, stds, covs = mc_predict(
                model,
                X_unlabeled,
                n_passes=mcfg.mc_passes,
                device=device,
            )
        t_pred_unlabeled = time.perf_counter() - t0

        means = means * y_std + y_mean
        stds = stds * y_std
        covs = covs * np.outer(y_std, y_std)[None, :, :]
        return means, stds, covs, t_pred_unlabeled

    t0 = time.perf_counter()
    with timed("predict_eval(unlabeled, no-uncertainty)"):
        means = predict_eval(model, X_unlabeled, device=device)
    t_pred_unlabeled = time.perf_counter() - t0

    means = means * y_std + y_mean
    stds = np.zeros_like(means, dtype=np.float32)
    covs = None
    return means, stds, covs, t_pred_unlabeled


@dataclass
class ALState:
    labeled_indices: list[int] = field(default_factory=list)
    unlabeled_indices: list[int] = field(default_factory=list)
    Y_labeled: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    hv_history: list[float] = field(default_factory=list)
    selections_per_iter: list[list[int]] = field(default_factory=list)
    acq_time_per_iter: list[float] = field(default_factory=list)
    train_metrics: list[dict] = field(default_factory=list)
    val_metrics: list[dict] = field(default_factory=list)
    sel_metrics: list[dict] = field(default_factory=list)
    iter_stage_times: list[dict] = field(default_factory=list)
    iteration: int = 0


def run_al_loop(
    X_pool,
    Y_pool,
    acq_fn: AcquisitionFunction,
    config: ExperimentConfig,
    seed: int,
    seed_indices: np.ndarray | None = None,
):
    N = len(Y_pool)
    n_obj = int(Y_pool.shape[1])
    al = config.al
    mcfg = config.model

    rng = np.random.default_rng(seed)
    select_params = set(inspect.signature(acq_fn.select).parameters.keys())

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

    state = ALState(
        labeled_indices=list(labeled),
        unlabeled_indices=list(unlabeled),
        Y_labeled=Y_pool[labeled].copy(),
    )

    state.selections_per_iter.append(list(labeled))

    hv0 = _hv(state.Y_labeled, al.ref_point)
    state.hv_history.append(hv0)
    state.acq_time_per_iter.append(0.0)
    state.train_metrics.append(_nan_metrics(n_obj))
    state.val_metrics.append(_nan_metrics(n_obj))
    state.sel_metrics.append(_nan_metrics(n_obj))

    LOGGER.info(
        f"[{acq_fn.name}] seed  | "
        f"labeled={len(state.labeled_indices):4d}  HV={hv0:.4f}  acq_time=0.000s"
    )

    for it in range(1, al.n_iterations + 1):
        state.iteration = it
        iter_t0 = time.perf_counter()

        t0 = time.perf_counter()
        with timed("build_model"):
            model = build_model(config.model, device=config.device, out_features=Y_pool.shape[1])
        t_build = time.perf_counter() - t0

        X_train = X_pool[state.labeled_indices]
        Y_train = state.Y_labeled

        Y_mean = Y_train.mean(axis=0)
        Y_std = np.maximum(Y_train.std(axis=0), 1e-8)
        Y_train_norm = (Y_train - Y_mean) / Y_std

        t0 = time.perf_counter()
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
        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        with timed("predict_eval(train)"):
            Y_train_pred = predict_eval(model, X_train, device=config.device)
        t_pred_train = time.perf_counter() - t0
        Y_train_pred = Y_train_pred * Y_std + Y_mean
        train_m = compute_regression_metrics(Y_train, Y_train_pred)
        state.train_metrics.append(train_m)

        X_unlabeled = X_pool[state.unlabeled_indices]

        needs_cov = getattr(acq_fn, "needs_full_cov", False)
        needs_uncertainty = getattr(acq_fn, "needs_uncertainty", True)

        means = None
        stds = None
        covs = None
        t_pred_unlabeled = 0.0
        t_cov_reconstruct = 0.0

        means, stds, covs, t_pred_unlabeled = _predict_unlabeled(
            model=model,
            X_unlabeled=X_unlabeled,
            mcfg=mcfg,
            device=config.device,
            needs_cov=needs_cov,
            needs_uncertainty=needs_uncertainty,
            y_mean=Y_mean,
            y_std=Y_std,
        )

        pareto_front = None
        hv_front = None

        if n_obj == 3 and state.Y_labeled.shape[0] > 0:
            with timed("pareto_front_max_3d_fast(labeled)"):
                pareto_front = pareto_front_max_3d_fast(state.Y_labeled)

            with timed("hypervolume_3d_max_fast(front)"):
                hv_front = hypervolume_3d_max_fast(pareto_front, al.ref_point)

            LOGGER.info(
                f"[PARETO] it={it} labeled={state.Y_labeled.shape[0]} "
                f"front={pareto_front.shape[0]} hv_front={hv_front:.6f}"
            )

        t0 = time.perf_counter()
        with timed("acquisition.select()"):
            cov_shape = None if covs is None else covs.shape
            std_shape = None if stds is None else stds.shape
            LOGGER.info(
                f"[ACQ] calling select "
                f"means={means.shape} stds={std_shape} covs={cov_shape}"
            )

            select_kwargs = {
                "k": al.batch_size,
                "covs": covs,
            }
            if "pareto_front" in select_params:
                select_kwargs["pareto_front"] = pareto_front
            if "pareto_hv" in select_params:
                select_kwargs["pareto_hv"] = hv_front

            sel_local = acq_fn.select(
                means,
                stds,
                state.Y_labeled,
                al.ref_point,
                **select_kwargs,
            )

        state.acq_time_per_iter.append(time.perf_counter() - t0)

        sel_local = np.asarray(sel_local).astype(int).ravel()
        U = len(state.unlabeled_indices)
        N_total = len(Y_pool)

        if sel_local.size == 0:
            raise RuntimeError("acq_fn.select returned empty selection")

        if sel_local.min() < 0 or sel_local.max() >= U:
            if sel_local.min() >= 0 and sel_local.max() < N_total:
                pos = {pool_idx: j for j, pool_idx in enumerate(state.unlabeled_indices)}
                try:
                    sel_local = np.array([pos[p] for p in sel_local], dtype=int)
                except KeyError as e:
                    raise RuntimeError(
                        f"acq_fn.select returned pool index not in unlabeled set: {e}"
                    ) from e
            else:
                raise RuntimeError(
                    f"sel_local indices out of range: min={sel_local.min()} max={sel_local.max()} "
                    f"(U={U}, N={N_total})"
                )

        sel_local = np.unique(sel_local)
        if sel_local.size > al.batch_size:
            sel_local = sel_local[-al.batch_size:]

        sel_pool = [state.unlabeled_indices[i] for i in sel_local]
        state.selections_per_iter.append(list(sel_pool))

        new_labels = Y_pool[sel_pool]

        sel_pred = means[sel_local]
        sel_m = compute_regression_metrics(new_labels, sel_pred)
        state.sel_metrics.append(sel_m)

        unlabeled_snapshot = list(state.unlabeled_indices)

        sel_pool_set = set(sel_pool)
        state.labeled_indices.extend(sel_pool)
        state.Y_labeled = np.vstack([state.Y_labeled, new_labels])
        state.unlabeled_indices = [i for i in state.unlabeled_indices if i not in sel_pool_set]

        if it % al.val_every == 0:
            Y_unlabeled_true = Y_pool[unlabeled_snapshot]
            val_m = compute_regression_metrics(Y_unlabeled_true, means)
        else:
            val_m = _nan_metrics(n_obj)
        state.val_metrics.append(val_m)

        t0 = time.perf_counter()
        with timed("hv(labeled_total)"):
            hv = _hv(state.Y_labeled, al.ref_point)
            hv_gain = hv - state.hv_history[-1]
        t_hv = time.perf_counter() - t0
        state.hv_history.append(hv)
        t_iter = time.perf_counter() - iter_t0

        state.iter_stage_times.append(
            {
                "total": float(t_iter),
                "build": float(t_build),
                "train": float(t_train),
                "pred_train": float(t_pred_train),
                "pred_unlab": float(t_pred_unlabeled),
                "cov_reconstruct": float(t_cov_reconstruct),
                "acq": float(state.acq_time_per_iter[-1]),
                "hv": float(t_hv),
            }
        )

        def _fmt_list(xs, fmt):
            return "[" + ",".join(format(float(x), fmt) for x in xs) + "]"

        tmse = _fmt_list(train_m["mse"], ".4f")
        tr2 = _fmt_list(train_m["r2"], ".2f")

        line = (
            f"[{acq_fn.name}] it {it:3d} | "
            f"labeled={len(state.labeled_indices):4d}  HV={hv:.4f}  (+{hv_gain:.4f})"
            f"  acq={state.acq_time_per_iter[-1]:.3f}s"
            f"  train_MSE={tmse} R2={tr2}"
        )

        if not np.isnan(val_m["mse"][0]):
            vmse = _fmt_list(val_m["mse"], ".4f")
            vr2 = _fmt_list(val_m["r2"], ".2f")
            line += f"  val_MSE={vmse} R2={vr2}"

        LOGGER.info(line)
        LOGGER.info(
            f"[TIME] [{acq_fn.name}] it {it:3d} | total={t_iter:.3f}s "
            f"build={t_build:.3f}s train={t_train:.3f}s pred_train={t_pred_train:.3f}s "
            f"pred_unlab={t_pred_unlabeled:.3f}s "
            f"cov_reconstruct={t_cov_reconstruct:.3f}s acq={state.acq_time_per_iter[-1]:.3f}s hv={t_hv:.3f}s"
        )

    _log_timing_summary(acq_fn.name, state)
    return state
