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
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._smiles)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._smiles), self._n_bits)

    def precompute(self, indices) -> None:
        """Cache fingerprints for the given pool indices.

        Only uncached indices are computed; already-cached ones are skipped.
        """
        uncached = [i for i in indices if i not in self._cache]
        if not uncached:
            return
        smiles = [self._smiles[i] for i in uncached]
        fps = compute_ecfp(smiles, radius=self._radius, n_bits=self._n_bits)
        for idx, row in zip(uncached, fps):
            self._cache[idx] = row

    def __getitem__(self, indices) -> np.ndarray:
        # Split into cached and uncached
        cached_positions = []
        uncached_positions = []
        uncached_smiles = []
        for pos, idx in enumerate(indices):
            if idx in self._cache:
                cached_positions.append(pos)
            else:
                uncached_positions.append(pos)
                uncached_smiles.append(self._smiles[idx])

        out = np.empty((len(indices), self._n_bits), dtype=np.float32)

        # Fill cached rows
        for pos in cached_positions:
            out[pos] = self._cache[indices[pos]]

        # Compute uncached rows on-the-fly (without caching)
        if uncached_smiles:
            fps = compute_ecfp(uncached_smiles, radius=self._radius, n_bits=self._n_bits)
            for i, pos in enumerate(uncached_positions):
                out[pos] = fps[i]

        return out
