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
