"""Interchangeable predictive-uncertainty backends for PAL."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import MLP, PiALMultilabelMLP, mc_predict, mc_predict_pial_counts


UNCERTAINTY_METHODS = ("mc_dropout", "last_layer_laplace")


@dataclass(frozen=True)
class LastLayerLaplaceState:
    """Independent Gaussian posteriors for rows of the final linear head."""

    parameter_covariances: np.ndarray


def _last_layer_features(
    model: MLP,
    X: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic head inputs (including bias) and MAP predictions."""
    model.eval()
    X_t = torch.from_numpy(np.asarray(X)).float()
    features: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            xb = X_t[start : start + batch_size].to(device, non_blocking=True)
            embedding = model.backbone(xb)
            predictions.append(model.head(embedding).cpu())
            ones = torch.ones((len(embedding), 1), dtype=embedding.dtype, device=embedding.device)
            features.append(torch.cat((embedding, ones), dim=1).cpu())

    if not features:
        width = int(model.head.in_features) + 1
        return (
            np.empty((0, width), dtype=np.float32),
            np.empty((0, model.out_features), dtype=np.float32),
        )

    return torch.cat(features).numpy(), torch.cat(predictions).numpy()


def fit_last_layer_laplace(
    model: MLP,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    prior_precision: float = 1.0,
    batch_size: int = 2048,
    device: str = "cpu",
) -> LastLayerLaplaceState:
    """Fit independent last-layer posteriors in the model's target space.

    ``Y`` must be in the same (normally z-score normalized) space in which the
    model was trained. The trained head remains the posterior mean. Only
    ``head.weight`` and ``head.bias`` are approximated; the backbone is fixed.
    """
    if isinstance(model, PiALMultilabelMLP):
        raise NotImplementedError(
            "Last-Layer Laplace is not yet implemented for PiAL multilabel heads"
        )
    if prior_precision <= 0:
        raise ValueError("prior_precision must be positive")

    phi, map_predictions = _last_layer_features(
        model, X, batch_size=batch_size, device=device
    )
    targets = np.asarray(Y, dtype=np.float64)
    if targets.ndim != 2 or targets.shape != map_predictions.shape:
        raise ValueError(f"Y must have shape {map_predictions.shape}, got {targets.shape}")
    if len(phi) == 0:
        raise ValueError("Cannot fit Last-Layer Laplace on an empty labeled set")

    phi64 = phi.astype(np.float64, copy=False)
    residuals = targets - map_predictions.astype(np.float64, copy=False)
    gram = phi64.T @ phi64
    eye = np.eye(phi64.shape[1], dtype=np.float64)
    covariances = np.empty(
        (targets.shape[1], phi64.shape[1], phi64.shape[1]), dtype=np.float64
    )

    for objective in range(targets.shape[1]):
        noise_variance = max(float(np.mean(residuals[:, objective] ** 2)), 1e-6)
        precision = prior_precision * eye + gram / noise_variance
        chol = np.linalg.cholesky(precision)
        inverse_chol = np.linalg.solve(chol, eye)
        covariances[objective] = inverse_chol.T @ inverse_chol

    return LastLayerLaplaceState(parameter_covariances=covariances)


def predict_with_uncertainty(
    model: MLP,
    X: np.ndarray,
    *,
    uncertainty_method: str = "mc_dropout",
    mc_passes: int = 50,
    batch_size: int = 2048,
    device: str = "cpu",
    laplace_state: LastLayerLaplaceState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, epistemic std and covariance in model target space."""
    if uncertainty_method == "mc_dropout":
        if isinstance(model, PiALMultilabelMLP):
            return mc_predict_pial_counts(
                model, X, n_passes=mc_passes, batch_size=batch_size, device=device
            )
        return mc_predict(
            model, X, n_passes=mc_passes, batch_size=batch_size, device=device
        )

    if uncertainty_method != "last_layer_laplace":
        raise ValueError(
            f"Unknown uncertainty_method={uncertainty_method!r}; expected one of {UNCERTAINTY_METHODS}"
        )
    if laplace_state is None:
        raise ValueError("laplace_state is required for last_layer_laplace prediction")

    phi, means = _last_layer_features(model, X, batch_size=batch_size, device=device)
    phi64 = phi.astype(np.float64, copy=False)
    posterior = laplace_state.parameter_covariances
    if posterior.shape[0] != model.out_features or posterior.shape[1] != phi.shape[1]:
        raise ValueError("laplace_state is incompatible with this model")

    variances = np.einsum("np,opq,nq->no", phi64, posterior, phi64, optimize=True)
    variances = np.maximum(variances, 0.0)
    stds = np.sqrt(variances).astype(np.float32)
    covs = np.zeros((len(phi), model.out_features, model.out_features), dtype=np.float32)
    diagonal = np.arange(model.out_features)
    covs[:, diagonal, diagonal] = variances.astype(np.float32)
    return means.astype(np.float32, copy=False), stds, covs
