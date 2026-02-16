import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from mlp_model import MLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32, copy=False)
        self.Y = Y.astype(np.float32, copy=False)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def load_labels(label_paths):
    dfs = []
    for p in label_paths:
        df = pd.read_parquet(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def compute_ecfp_from_smiles(smiles_list, radius=2, n_bits=2048):
    """
    Returns np.float32 array [N, n_bits] with 0/1 bits (Morgan/ECFP).
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except Exception as e:
        raise RuntimeError(
            "RDKit nie jest dostępny w tym środowisku. "
            "Zainstaluj rdkit albo powiedz — dam wersję z wczytywaniem X z osobnego pliku."
        ) from e

    X = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    bad = 0
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad += 1
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=True)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        X[i] = arr.astype(np.float32)
    if bad:
        logging.warning("RDKit: %d/%d SMILES nie dało się sparsować (X będzie zerowe dla tych wierszy).", bad, len(smiles_list))
    return X


def ensure_smiles_and_targets(df: pd.DataFrame, pool_parquet: str, y_cols: tuple) -> pd.DataFrame:
    """
    Ensures df has 'smiles' and y_cols by joining from pool_parquet (safe columns only).
    DOES NOT read X_ecfp_2 from pool (avoids Arrow overflow).
    """
    need = []
    if "smiles" not in df.columns:
        need.append("smiles")
    for yc in y_cols:
        if yc not in df.columns:
            need.append(yc)

    if not need:
        return df

    if pool_parquet is None:
        raise ValueError(f"Brakuje kolumn {need} w label parquet, a nie podano --pool_parquet.")

    # Safe read: only scalar columns (no list columns)
    pool_cols = ["ID", *need]
    pool = pd.read_parquet(pool_parquet, columns=pool_cols)

    df = df.copy()
    df["ID"] = df["ID"].astype(str)
    pool["ID"] = pool["ID"].astype(str)

    merged = df.merge(pool, on="ID", how="left")

    miss_cols = [c for c in need if merged[c].isna().any()]
    if miss_cols:
        bad = merged[merged[miss_cols].isna().any(axis=1)]["ID"].head(10).tolist()
        raise RuntimeError(
            f"Po merge z pool brakuje wartości w kolumnach {miss_cols} dla części ID. "
            f"Przykładowe ID: {bad}"
        )

    return merged


def build_xy(df: pd.DataFrame, y_cols: tuple, negate_targets: bool, radius: int, n_bits: int):
    # y
    Y = df[list(y_cols)].values.astype(np.float32)
    if negate_targets:
        Y = -Y

    # X from smiles (RDKit)
    X = compute_ecfp_from_smiles(df["smiles"].astype(str).tolist(), radius=radius, n_bits=n_bits)
    return X, Y


def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, loss_fn):
    model.eval()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_paths", nargs="+", required=True,
                    help="np. data/labels/iter_000.parquet data/labels/iter_001.parquet ...")
    ap.add_argument("--pool_parquet", default=None,
                    help="np. data/pool/savi_merged_all.parquet (wymagane jeśli label parquet nie ma smiles/y)")
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--meta_out", default=None)

    ap.add_argument("--y_cols", nargs=2, default=["score_3GVB", "score_6D6P"])
    ap.add_argument("--negate_targets", action="store_true", help="negate docking so higher=better")

    ap.add_argument("--ecfp_radius", type=int, default=2)
    ap.add_argument("--ecfp_bits", type=int, default=2048)

    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--num_hidden_layers", type=int, default=3)
    ap.add_argument("--dropout_rate", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    args = ap.parse_args()

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    y_cols = tuple(args.y_cols)

    logging.info("Loading labels: %s", args.label_paths)
    df = load_labels(args.label_paths)

    if "ID" not in df.columns:
        raise ValueError("Brak kolumny 'ID' w label parquet.")
    if "split" not in df.columns:
        raise ValueError("Brak kolumny 'split' w label parquet.")

    # Ensure smiles and y are present (safe columns only)
    df = ensure_smiles_and_targets(df, args.pool_parquet, y_cols)

    # Split: train/val/test
    split = df["split"].astype(str)

    # val/test tylko z pierwszej iteracji (jeśli istnieją)
    df_val = df[split == "val"].copy()
    df_test = df[split == "test"].copy()

    # WSZYSTKO inne → TRAIN
    df_train = df[(split != "val") & (split != "test")].copy()
    logging.info("Rows: train=%d val=%d test=%d", len(df_train), len(df_val), len(df_test))

    # Build X/Y per split (RDKit)
    Xtr, Ytr = build_xy(df_train, y_cols, bool(args.negate_targets), args.ecfp_radius, args.ecfp_bits)
    Xva, Yva = build_xy(df_val, y_cols, bool(args.negate_targets), args.ecfp_radius, args.ecfp_bits)
    Xte, Yte = build_xy(df_test, y_cols, bool(args.negate_targets), args.ecfp_radius, args.ecfp_bits)

    ds_train = NumpyDataset(Xtr, Ytr)
    ds_val = NumpyDataset(Xva, Yva)
    ds_test = NumpyDataset(Xte, Yte)

    in_features = Xtr.shape[1]

    model = MLP(
        in_features=in_features,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        dropout_rate=args.dropout_rate,
        out_features=2,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, loss_fn)
        va = eval_loss(model, val_loader, loss_fn)
        logging.info("Epoch %d/%d | train=%.6f val=%.6f", ep, args.epochs, tr, va)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    te = eval_loss(model, test_loader, loss_fn)
    logging.info("Best val=%.6f | test=%.6f", best_val, te)

    ckpt = {
        "in_features": in_features,
        "hparams": {
            "hidden_dim": args.hidden_dim,
            "num_hidden_layers": args.num_hidden_layers,
            "dropout_rate": args.dropout_rate,
            "negate_targets": bool(args.negate_targets),
            "y_cols": list(y_cols),
            "ecfp_radius": args.ecfp_radius,
            "ecfp_bits": args.ecfp_bits,
        },
        "model_state_dict": best_state,
        "metrics": {"best_val_mse": best_val, "test_mse": te},
    }
    torch.save(ckpt, out_ckpt)
    logging.info("Saved ckpt -> %s", str(out_ckpt))

    if args.meta_out:
        meta = {
            "label_paths": args.label_paths,
            "pool_parquet": args.pool_parquet,
            "rows": {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
            "metrics": ckpt["metrics"],
            "hparams": ckpt["hparams"],
        }
        Path(args.meta_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.meta_out).write_text(json.dumps(meta, indent=2))
        logging.info("Saved meta -> %s", args.meta_out)


if __name__ == "__main__":
    main()
