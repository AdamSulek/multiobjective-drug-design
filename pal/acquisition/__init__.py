"""Acquisition function registry and factory."""

from .base import AcquisitionFunction
from .ellipse_fast import FastEllipseAcquisition
from .random import RandomAcquisition
from .ucb import UCBExplorationAcquisition
from .ellipse_direction import EllipseDirectionAcquisition
from .ucb_flexible import UCBExplorationAcquisitionFlexible
from .ellipse_fast_flexible import FastEllipseAcquisitionFlexible
from .ellipse_direction_flexible import EllipseDirectionAcquisitionFlexible


REGISTRY: dict[str, type[AcquisitionFunction]] = {
    "ucb": UCBExplorationAcquisition,
    "ucb_flexible": UCBExplorationAcquisitionFlexible,
    "random": RandomAcquisition,
    "ellipse_fast": FastEllipseAcquisition,
    "ellipse_fast_flexible": FastEllipseAcquisitionFlexible,
    "ellipse_directions": EllipseDirectionAcquisition,
    "ellipse_directions_flexible": EllipseDirectionAcquisitionFlexible,
}


def get_acquisition(name: str, **kwargs) -> AcquisitionFunction:
    """Instantiate an acquisition function by name.

    Parameters
    ----------
    name : str
        Key in ``REGISTRY`` (e.g. ``"exploitation"``, ``"ucb"``, ``"random"``).
    **kwargs
        Forwarded to the constructor (e.g. ``k_ucb=2.0`` for UCB).

    Returns
    -------
    AcquisitionFunction
    """
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown acquisition function '{name}'. "
            f"Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[name](**kwargs)
