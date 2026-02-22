"""Acquisition function registry and factory."""

from .base import AcquisitionFunction
from .ellipse import EllipseAcquisition
from .ellipse_fast import FastEllipseAcquisition
from .exploitation import ExploitationAcquisition
from .random import RandomAcquisition
from .ucb import UCBExplorationAcquisition

REGISTRY: dict[str, type[AcquisitionFunction]] = {
    "exploitation": ExploitationAcquisition,
    "ucb": UCBExplorationAcquisition,
    "random": RandomAcquisition,
    "ellipse": EllipseAcquisition,
    "ellipse_fast": FastEllipseAcquisition,
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
