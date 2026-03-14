#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

Y_COLS = ("score_3GVB", "score_6D6P")


# ======================================================
# ECFP
# ======================================================
def ecfp2_bits_from_smiles(smiles_list, nbits=2048, use_chirality=False):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    out = []
    bad_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(None)
            bad_idx.append(i)
            continue

        bv = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=nbits, useChirality=use_chirality
        )
        arr = np.zeros((nbits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bv, arr)
        out.append(arr)

    if bad_idx:
        ex = [(int(i), smiles_list[int(i)]) for i in bad_idx[:5]]
        raise RuntimeError(
            f"RDKit MolFromSmiles failed for {len(bad_idx)} SMILES. Examples (idx,smiles): {ex}"
        )

    return out


# ======================================================
# MAIN
# ======================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected_csv", required=True)
    ap.add_argument("--pool_parquet", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--ecfp_bits", type=int, default=2048)
    ap.add_argument("--no_chirality", action="store_true")
    args = ap.parse_args()

    selected_csv = Path(args.selected_csv)
    pool_parquet = Path(args.pool_parquet)
    out_parquet = Path(args.out_parquet)

    # ======================================================
    # SELECTED → ID
    # ======================================================
    sel = pd.read_csv(selected_csv)
    if "ID" in sel.columns:
        sel_ids = sel[["ID"]].drop_duplicates().copy()
    elif "mol_id" in sel.columns:
        sel_ids = sel[["mol_id"]].drop_duplicates().rename(columns={"mol_id": "ID"}).copy()
    else:
        raise ValueError(f"{selected_csv} must contain column 'ID' or 'mol_id'")

    if sel_ids.empty:
        raise RuntimeError(f"{selected_csv} has 0 rows after drop_duplicates()")

    sel_ids["ID"] = sel_ids["ID"].astype(str)

    # ======================================================
    # POOL → tylko scalar kolumny
    # ======================================================
    need_cols = ["ID", args.smiles_col, *Y_COLS]
    pool = pd.read_parquet(pool_parquet, columns=need_cols)
    pool["ID"] = pool["ID"].astype(str)

    merged = sel_ids.merge(pool, on="ID", how="left")

    miss_basic = merged[[args.smiles_col, *Y_COLS]].isna().any(axis=1)
    if miss_basic.any():
        bad = merged.loc[miss_basic, "ID"].head(10).tolist()
        raise RuntimeError(
            f"After merge, missing smiles/y for {int(miss_basic.sum())} rows. Example IDs: {bad}"
        )

    # ======================================================
    # ECFP
    # ======================================================
    logging.info(
        "Generating X_ecfp_2 from SMILES (nbits=%d, chirality=%s) for %d molecules",
        args.ecfp_bits,
        str(not args.no_chirality),
        len(merged),
    )

    merged["X_ecfp_2"] = ecfp2_bits_from_smiles(
        merged[args.smiles_col].tolist(),
        nbits=args.ecfp_bits,
        use_chirality=(not args.no_chirality),
    )

    lens = merged["X_ecfp_2"].map(len).unique().tolist()
    if lens != [args.ecfp_bits]:
        raise RuntimeError(
            f"Unexpected fingerprint lengths: {lens} (expected {args.ecfp_bits})"
        )

    # ======================================================
    # 🔥 KLUCZOWA ZMIANA
    # ======================================================
    merged["split"] = "train"

    # ======================================================
    # SAVE
    # ======================================================
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_parquet, index=False)

    logging.info(
        "Saved -> %s (rows=%d) | split=train added",
        out_parquet,
        len(merged),
    )


if __name__ == "__main__":
    main()
