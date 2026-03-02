"""SMILES -> ECFP (Morgan) fingerprint featurization."""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def compute_ecfp(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Convert a list of SMILES to ECFP fingerprints.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings.
    radius : int
        Morgan fingerprint radius (2 = ECFP4).
    n_bits : int
        Length of the bit vector.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(N, n_bits)`` with 0/1 bits.
    """
    X = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits, useChirality=True
        )
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        X[i] = arr.astype(np.float32)
    return X


class LazyECFP:
    """Lazy fingerprint array that computes ECFP on-demand when indexed.

    Supports ``len()``, ``.shape``, and ``[]`` fancy indexing with a list/array
    of ints — the only access patterns used by the AL loop.
    """

    def __init__(
        self,
        smiles_list: list[str],
        radius: int = 2,
        n_bits: int = 2048,
    ) -> None:
        self._smiles = smiles_list
        self._radius = radius
        self._n_bits = n_bits

    def __len__(self) -> int:
        return len(self._smiles)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._smiles), self._n_bits)

    def __getitem__(self, indices) -> np.ndarray:
        subset = [self._smiles[i] for i in indices]
        return compute_ecfp(subset, radius=self._radius, n_bits=self._n_bits)
