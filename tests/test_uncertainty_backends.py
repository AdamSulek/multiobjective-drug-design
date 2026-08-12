import unittest

import numpy as np
import torch

from pal.acquisition.ellipse_direction import EllipseDirectionAcquisition
from pal.acquisition.ellipse_fast import FastEllipseAcquisition
from pal.acquisition.ucb import UCBExplorationAcquisition
from pal.config import ModelConfig
from pal.model import build_model, train_model
from pal.uncertainty import fit_last_layer_laplace, predict_with_uncertainty


class UncertaintyBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(17)
        torch.manual_seed(17)
        X = rng.normal(size=(42, 8)).astype(np.float32)
        coefficients = rng.normal(size=(8, 3)).astype(np.float32)
        Y_raw = X @ coefficients + 0.05 * rng.normal(size=(42, 3)).astype(np.float32)
        cls.X_train, cls.X_candidates = X[:32], X[32:]
        cls.Y_train = Y_raw[:32]
        cls.y_mean = cls.Y_train.mean(axis=0)
        cls.y_std = np.maximum(cls.Y_train.std(axis=0), 1e-8)
        Y_train_norm = (cls.Y_train - cls.y_mean) / cls.y_std

        cfg = ModelConfig(
            in_features=8, hidden_sizes=(12,), dropout=0.25, out_features=3,
            epochs=12, batch_size=16, patience=0,
        )
        cls.model = build_model(cfg, out_features=3)
        train_model(
            cls.model, cls.X_train, Y_train_norm, epochs=cfg.epochs,
            batch_size=cfg.batch_size, patience=0, device="cpu",
        )
        cls.laplace_state = fit_last_layer_laplace(
            cls.model, cls.X_train, Y_train_norm,
            prior_precision=1.0, batch_size=16,
        )

        cls.outputs = {}
        for method in ("mc_dropout", "last_layer_laplace"):
            means_norm, stds_norm, covs_norm = predict_with_uncertainty(
                cls.model,
                cls.X_candidates,
                uncertainty_method=method,
                mc_passes=12,
                batch_size=10,
                laplace_state=cls.laplace_state,
            )
            # Identical denormalization for both uncertainty backends.
            means = means_norm * cls.y_std + cls.y_mean
            stds = stds_norm * cls.y_std
            covs = covs_norm * np.outer(cls.y_std, cls.y_std)[None, :, :]
            cls.outputs[method] = (means, stds, covs)

    def test_backend_contract_and_values(self):
        for means, stds, covs in self.outputs.values():
            self.assertEqual(means.shape, (10, 3))
            self.assertEqual(stds.shape, (10, 3))
            self.assertEqual(covs.shape, (10, 3, 3))
            self.assertTrue(np.isfinite(means).all())
            self.assertTrue(np.isfinite(stds).all())
            self.assertTrue(np.isfinite(covs).all())
            self.assertTrue((stds >= 0).all())

    def test_laplace_is_deterministic_and_diagonal(self):
        kwargs = dict(
            uncertainty_method="last_layer_laplace",
            laplace_state=self.laplace_state,
            batch_size=10,
        )
        first = predict_with_uncertainty(self.model, self.X_candidates, **kwargs)
        second = predict_with_uncertainty(self.model, self.X_candidates, **kwargs)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        off_diagonal = first[2].copy()
        off_diagonal[:, np.arange(3), np.arange(3)] = 0.0
        self.assertEqual(np.count_nonzero(off_diagonal), 0)

    def test_acquisitions_accept_both_backends_unchanged(self):
        acquisitions = (
            UCBExplorationAcquisition(k_ucb=1.0),
            FastEllipseAcquisition(k=1.0, n_directions=8),
            EllipseDirectionAcquisition(k=1.0, n_directions=8),
        )
        ref_point = tuple((self.Y_train.min(axis=0) - 1.0).tolist())
        for acquisition in acquisitions:
            for means, stds, covs in self.outputs.values():
                selected = acquisition.select(
                    means, stds, self.Y_train, ref_point, k=2, covs=covs
                )
                self.assertEqual(selected.shape, (2,))

    def test_print_ten_candidate_comparison(self):
        for method, (means, stds, _) in self.outputs.items():
            print(f"\n{method}: dock_mean dock_std selected_mean selected_std remaining_mean remaining_std")
            for mean, std in zip(means, stds):
                values = (mean[0], std[0], mean[1], std[1], mean[2], std[2])
                print(" ".join(f"{value:.6f}" for value in values))


if __name__ == "__main__":
    unittest.main()
