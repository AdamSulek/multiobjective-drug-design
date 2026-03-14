"""ZINC dataset with SA and QED oracles."""

import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, RDConfig

# SA scorer lives in RDKit's Contrib directory
sys.path.append(str(RDConfig.RDContribDir) + "/SA_Score")
import sascorer  # noqa: E402


def load_zinc_smiles(n: int = 5000, seed: int = 42) -> list[str]:
    """Load *n* random SMILES from ZINC via TDC MolGen."""
    from tdc.generation import MolGen

    data = MolGen(name="ZINC")
    df = data.get_data()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    return df.iloc[idx]["smiles"].tolist()


def compute_sa_score(smiles: str) -> float:
    """Compute raw SA score (1=easy .. 10=hard) for a single SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")
    return sascorer.calculateScore(mol)


def compute_qed_score(smiles: str) -> float:
    """Compute QED drug-likeness score (0..1, higher=better)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")
    return QED.qed(mol)


def generate_zinc_dataset(
    n_compounds: int = 5000, seed: int = 42
) -> pd.DataFrame:
    """Generate the benchmark dataset.

    Returns a DataFrame with columns:
        smiles, sa_raw, qed, sa_score

    where ``sa_score = 10 - sa_raw`` so that **both** objectives are
    to be **maximized**.
    """
    smiles_list = load_zinc_smiles(n=n_compounds, seed=seed)

    sa_raw = np.array([compute_sa_score(s) for s in smiles_list])
    qed = np.array([compute_qed_score(s) for s in smiles_list])
    sa_score = 10.0 - sa_raw  # higher = more synthesizable

    df = pd.DataFrame(
        {
            "smiles": smiles_list,
            "sa_raw": sa_raw,
            "qed": qed,
            "sa_score": sa_score,
        }
    )
    # drop rows where either oracle failed
    df = df.dropna(subset=["sa_raw", "qed"]).reset_index(drop=True)
    return df


def load_dataset_from_file(
    path: str,
    property_cols: list[str],
    smiles_col: str | None = None,
    fingerprint_col: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Load a dataset from a CSV or parquet file.

    Parameters
    ----------
    path : str
        Path to CSV or parquet file.
    property_cols : list[str]
        2 or 3 column names for the objectives.
    smiles_col : str or None
        Column containing SMILES strings (caller computes ECFP).
    fingerprint_col : str or None
        Column containing precomputed fingerprint arrays.

    Returns
    -------
    (df, X) where X is the precomputed fingerprint matrix or None.
    """
    if len(property_cols) not in (2, 3):
        raise ValueError(f"2 or 3 property columns required, got {len(property_cols)}")
    
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "parquet":
        df = pd.read_parquet(path)
    elif ext in ("csv", "tsv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension: .{ext}")

    missing = [c for c in property_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    df = df.dropna(subset=property_cols).reset_index(drop=True)

    if fingerprint_col is not None:
        if fingerprint_col not in df.columns:
            raise ValueError(f"Fingerprint column '{fingerprint_col}' not found")
        X = np.stack(df[fingerprint_col].values).astype(np.float32)
        return df, X

    if smiles_col is not None:
        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found")
        return df, None

    raise ValueError("Either smiles_col or fingerprint_col must be provided")


def oracle(smiles_list: list[str]) -> np.ndarray:
    """Compute ground-truth objective values for a list of SMILES.

    Returns
    -------
    np.ndarray
        Shape ``(N, 2)`` array of ``[sa_score, qed]``.
    """
    sa_raw = np.array([compute_sa_score(s) for s in smiles_list])
    qed = np.array([compute_qed_score(s) for s in smiles_list])
    sa_score = 10.0 - sa_raw
    return np.column_stack([sa_score, qed])
