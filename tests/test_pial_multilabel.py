import os
import unittest

import numpy as np
import torch

from pal.model import (
    PiALMultilabelMLP,
    mc_predict_pial_counts,
    predict_pial_outputs,
    train_pial_multilabel_model,
)
from pal.pial_data import (
    interaction_vectors_to_binary,
    load_pial_training_rows,
    load_plif_mapping,
    validate_binary_counts,
)


class PiALMultilabelTest(unittest.TestCase):
    def test_mapping_and_small_binary_counts(self):
        mapping = load_plif_mapping()
        self.assertEqual(len(mapping.selected_aggregated), 8)
        self.assertEqual(len(mapping.remaining_aggregated), 48)
        selected, remaining = interaction_vectors_to_binary(
            [[0, 1, 3, 4], [21, 22, 23, 73, 74], []], mapping
        )
        self.assertEqual(selected.shape, (3, 8))
        self.assertEqual(remaining.shape, (3, 48))
        np.testing.assert_array_equal(selected.sum(1), [1, 2, 0])
        np.testing.assert_array_equal(remaining.sum(1), [2, 1, 0])

    @unittest.skipUnless(
        os.environ.get("PIAL_VALIDATE_FULL_COUNTS") == "1",
        "set PIAL_VALIDATE_FULL_COUNTS=1 for the full shared-data check",
    )
    def test_binary_sums_match_all_pareto_counts(self):
        result = validate_binary_counts()
        self.assertEqual(result["n_rows"], 4_995_876)
        self.assertEqual(result["n_selected"], 8)
        self.assertEqual(result["n_remaining"], 48)

    def test_indexed_shared_data_loader(self):
        rows = load_pial_training_rows([0, 2, 1])
        self.assertEqual(rows["X"].shape, (3, 2048))
        self.assertEqual(rows["docking"].shape, (3,))
        self.assertEqual(rows["selected"].shape, (3, 8))
        self.assertEqual(rows["remaining"].shape, (3, 48))
        np.testing.assert_allclose(rows["docking"], [7.6, 8.1, 7.9])
        np.testing.assert_array_equal(rows["selected"].sum(1), [1, 2, 2])
        np.testing.assert_array_equal(rows["remaining"].sum(1), [3, 2, 2])

    def test_outputs_losses_probabilities_and_acquisition_counts(self):
        rng = np.random.default_rng(7)
        torch.manual_seed(7)
        n, width = 32, 16
        X = rng.normal(size=(n, width)).astype(np.float32)
        docking = rng.normal(size=n).astype(np.float32)
        selected = rng.integers(0, 2, size=(n, 8)).astype(np.float32)
        remaining = rng.integers(0, 2, size=(n, 48)).astype(np.float32)
        model = PiALMultilabelMLP(
            in_features=width, hidden_sizes=(24, 12), dropout=0.1
        )

        raw = model(torch.from_numpy(X[:5]))
        self.assertEqual(tuple(raw[0].shape), (5, 1))
        self.assertEqual(tuple(raw[1].shape), (5, 8))
        self.assertEqual(tuple(raw[2].shape), (5, 48))

        losses = train_pial_multilabel_model(
            model, X, docking, selected, remaining,
            epochs=3, batch_size=8, lr=1e-3,
        )
        for name in ("docking", "selected", "remaining", "total"):
            self.assertTrue(np.isfinite(losses[name]))
            self.assertGreaterEqual(losses[name], 0.0)

        dock, selected_probs, remaining_probs, selected_count, remaining_count = (
            predict_pial_outputs(model, X[:5], batch_size=5)
        )
        self.assertEqual(dock.shape, (5, 1))
        self.assertEqual(selected_probs.shape, (5, 8))
        self.assertEqual(remaining_probs.shape, (5, 48))
        self.assertEqual(selected_count.shape, (5,))
        self.assertEqual(remaining_count.shape, (5,))
        np.testing.assert_allclose(selected_count, selected_probs.sum(1))
        np.testing.assert_allclose(remaining_count, remaining_probs.sum(1))
        self.assertTrue(((selected_probs >= 0) & (selected_probs <= 1)).all())
        self.assertTrue(((remaining_probs >= 0) & (remaining_probs <= 1)).all())

        means, stds, covs = mc_predict_pial_counts(
            model, X[:5], n_passes=3, batch_size=5
        )
        self.assertEqual(means.shape, (5, 3))
        self.assertEqual(stds.shape, (5, 3))
        self.assertEqual(covs.shape, (5, 3, 3))
        print("SANITY_LOSSES", losses)
        print("OUTPUT_SHAPES", raw[0].shape, raw[1].shape, raw[2].shape)
        print("PROBABILITY_SHAPES", selected_probs.shape, remaining_probs.shape)


if __name__ == "__main__":
    unittest.main()
