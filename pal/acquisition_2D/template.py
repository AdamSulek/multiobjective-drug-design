"""Copy-paste template for implementing a new acquisition function.

How to create a new acquisition strategy
=========================================

1. Copy this file and rename it (e.g. ``my_strategy.py``).
2. Rename the class and fill in ``name`` and ``score``.
3. Register it in ``pal/acquisition/__init__.py``::

       from .my_strategy import MyStrategyAcquisition
       REGISTRY["my_strategy"] = MyStrategyAcquisition

4. Run it::

       python -m pal.compare --strategies exploitation,ucb,my_strategy

Available utilities you can import
-----------------------------------
- ``from pal.pareto import pareto_front_2d``
      Compute the 2-D Pareto front (maximization). Returns (M, 2) array.

- ``from pal.pareto import hypervolume_2d``
      Compute dominated hypervolume given points and a reference point.

- ``from pal.pareto import delta_hv_contribution``
      HV improvement of a single candidate point (x, y) against the front.

- ``from pal.pareto import stair_y_at_x``
      Baseline y-height of the current stair at coordinate x.

- ``from pal.pareto import build_stair_polygon``
      Polygon vertices for plotting the dominated region.
"""

from typing import Tuple

import numpy as np

from ..pareto import delta_hv_contribution, pareto_front_2d
from .base import AcquisitionFunction


class TemplateAcquisition(AcquisitionFunction):
    """<One-line description of your strategy>."""

    def __init__(self, my_param: float = 1.0):
        self.my_param = my_param

    @property
    def name(self) -> str:
        return f"Template(p={self.my_param})"

    def score(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        current_labels: np.ndarray,
        ref_point: Tuple[float, float],
    ) -> np.ndarray:
        """Return one score per candidate (higher = pick first).

        Parameters
        ----------
        means : (N, 2)  predicted objective means
        stds  : (N, 2)  predicted objective std devs
        current_labels : (M, 2)  objectives of already-labeled compounds
        ref_point : reference point for HV computation

        Returns
        -------
        scores : (N,)
        """
        front = pareto_front_2d(current_labels)

        scores = np.zeros(len(means), dtype=np.float32)
        for i in range(len(means)):
            # --- replace the body below with your logic ---
            scores[i] = delta_hv_contribution(
                front, means[i, 0], means[i, 1], ref_point
            )
        return scores
