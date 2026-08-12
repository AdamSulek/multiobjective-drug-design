"""Read-only reconstruction of PiAL multilabel targets from sparse PLIF data."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_DIMENSIONS_PATH = Path("/net/storage/pr3/plgrid/plggsanodrugs/pial/data/pareto_dimensions.parquet")
DEFAULT_FULL_PLIF_PATH = Path("/net/storage/pr3/plgrid/plggsanodrugs/pial/docking/data/ampc_1L2S_B_training_PLIF.parquet")
DEFAULT_MAPPING_PATH = Path("/net/storage/pr3/plgrid/plggsanodrugs/pial/data/pareto_plif_feature_mapping.tsv")


@dataclass(frozen=True)
class PiALPLIFMapping:
    original_to_aggregated: np.ndarray
    aggregated_names: tuple[str, ...]
    selected_aggregated: tuple[int, ...]
    remaining_aggregated: tuple[int, ...]
    aggregated_to_selected: np.ndarray
    aggregated_to_remaining: np.ndarray


def load_plif_mapping(path: str | Path = DEFAULT_MAPPING_PATH) -> PiALPLIFMapping:
    """Load and validate the canonical 75 -> 56 -> (8, 48) mapping."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    original = np.asarray([int(row["original_feature_idx"]) for row in rows])
    if not np.array_equal(original, np.arange(len(rows))):
        raise ValueError("original_feature_idx must be contiguous and zero-based")
    original_to_aggregated = np.asarray(
        [int(row["aggregated_feature_idx"]) for row in rows], dtype=np.int64
    )
    n_aggregated = int(original_to_aggregated.max()) + 1
    names = [""] * n_aggregated
    selected_flags = np.zeros(n_aggregated, dtype=bool)
    remaining_flags = np.zeros(n_aggregated, dtype=bool)
    for row, aggregated in zip(rows, original_to_aggregated):
        names[aggregated] = row["aggregated_feature_name"]
        selected_flags[aggregated] |= row["is_selected_interaction"].lower() == "true"
        remaining_flags[aggregated] |= row["is_remaining_interaction"].lower() == "true"
    if np.any(selected_flags & remaining_flags) or not np.all(selected_flags | remaining_flags):
        raise ValueError("Aggregated features must be exclusively selected or remaining")
    selected = tuple(np.flatnonzero(selected_flags).tolist())
    remaining = tuple(np.flatnonzero(remaining_flags).tolist())
    if (len(rows), n_aggregated, len(selected), len(remaining)) != (75, 56, 8, 48):
        raise ValueError("Expected PiAL mapping 75 -> 56 -> (8 selected, 48 remaining)")
    to_selected = np.full(n_aggregated, -1, dtype=np.int16)
    to_remaining = np.full(n_aggregated, -1, dtype=np.int16)
    to_selected[list(selected)] = np.arange(8)
    to_remaining[list(remaining)] = np.arange(48)
    return PiALPLIFMapping(
        original_to_aggregated, tuple(names), selected, remaining,
        to_selected, to_remaining,
    )


def interaction_vectors_to_binary(
    interaction_vectors: Iterable[Sequence[int] | None],
    mapping: PiALPLIFMapping,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sparse original indices into selected-8 and remaining-48 labels."""
    vectors = list(interaction_vectors)
    selected = np.zeros((len(vectors), 8), dtype=np.float32)
    remaining = np.zeros((len(vectors), 48), dtype=np.float32)
    for row_index, source_indices in enumerate(vectors):
        if source_indices is None or len(source_indices) == 0:
            continue
        source = np.asarray(source_indices, dtype=np.int64)
        if source.min() < 0 or source.max() >= len(mapping.original_to_aggregated):
            raise ValueError(f"Invalid PLIF source index in row {row_index}")
        aggregated = np.unique(mapping.original_to_aggregated[source])
        selected_local = mapping.aggregated_to_selected[aggregated]
        remaining_local = mapping.aggregated_to_remaining[aggregated]
        selected[row_index, selected_local[selected_local >= 0]] = 1.0
        remaining[row_index, remaining_local[remaining_local >= 0]] = 1.0
    return selected, remaining


def validate_binary_counts(
    full_plif_path: str | Path = DEFAULT_FULL_PLIF_PATH,
    dimensions_path: str | Path = DEFAULT_DIMENSIONS_PATH,
    mapping_path: str | Path = DEFAULT_MAPPING_PATH,
    batch_size: int = 100_000,
) -> dict[str, int]:
    """Stream aligned Parquets and verify binary sums against stored counts."""
    import pyarrow.parquet as pq

    mapping = load_plif_mapping(mapping_path)
    source = pq.ParquetFile(full_plif_path)
    dimensions = pq.ParquetFile(dimensions_path)
    if source.metadata.num_rows != dimensions.metadata.num_rows:
        raise ValueError("Full PLIF and dimensions row counts differ")
    checked = 0
    source_batches = source.iter_batches(batch_size=batch_size, columns=["ID", "interaction_vector"])
    dimension_batches = dimensions.iter_batches(
        batch_size=batch_size,
        columns=["molecule_index", "molecule_id", "selected_interactions", "remaining_interactions"],
    )
    for source_batch, dimension_batch in zip(source_batches, dimension_batches):
        source_ids = np.asarray(source_batch.column("ID").to_pylist(), dtype=object)
        dimension_ids = np.asarray(dimension_batch.column("molecule_id").to_pylist(), dtype=object)
        if not np.array_equal(source_ids, dimension_ids):
            raise ValueError(f"Molecule ID mismatch at row {checked}")
        indices = dimension_batch.column("molecule_index").to_numpy()
        if not np.array_equal(indices, np.arange(checked, checked + len(indices))):
            raise ValueError(f"molecule_index mismatch at row {checked}")
        selected, remaining = interaction_vectors_to_binary(
            source_batch.column("interaction_vector").to_pylist(), mapping
        )
        selected_counts = dimension_batch.column("selected_interactions").to_numpy()
        remaining_counts = dimension_batch.column("remaining_interactions").to_numpy()
        if not np.array_equal(selected.sum(1), selected_counts):
            raise AssertionError(f"selected binary/count mismatch at row {checked}")
        if not np.array_equal(remaining.sum(1), remaining_counts):
            raise AssertionError(f"remaining binary/count mismatch at row {checked}")
        checked += source_batch.num_rows
    if checked != source.metadata.num_rows:
        raise RuntimeError(f"Validated {checked} of {source.metadata.num_rows} rows")
    return {"n_rows": checked, "n_selected": 8, "n_remaining": 48}


def load_pial_training_rows(
    molecule_indices: Sequence[int],
    *,
    h5_path: str | Path = "/net/storage/pr3/plgrid/plggsanodrugs/pial/data/dataset.h5",
    full_plif_path: str | Path = DEFAULT_FULL_PLIF_PATH,
    mapping_path: str | Path = DEFAULT_MAPPING_PATH,
    batch_size: int = 100_000,
) -> dict[str, np.ndarray]:
    """Load only requested ECFP/docking rows and reconstruct their 8+48 labels."""
    import h5py
    import pyarrow.parquet as pq

    requested = np.asarray(molecule_indices, dtype=np.int64)
    if requested.ndim != 1 or len(requested) == 0:
        raise ValueError("molecule_indices must be a non-empty one-dimensional sequence")
    if len(np.unique(requested)) != len(requested):
        raise ValueError("molecule_indices must not contain duplicates")

    order = np.argsort(requested)
    sorted_indices = requested[order]
    with h5py.File(h5_path, "r") as h5:
        n_rows = len(h5["neg_docking_score"])
        if sorted_indices[0] < 0 or sorted_indices[-1] >= n_rows:
            raise IndexError(f"molecule index outside [0, {n_rows - 1}]")
        packed_sorted = h5["X_ecfp2_packed"][sorted_indices]
        docking_sorted = h5["neg_docking_score"][sorted_indices].astype(np.float32)
        ids_sorted = np.asarray(h5["ID"][sorted_indices], dtype=object)

    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    packed = packed_sorted[inverse]
    docking = docking_sorted[inverse]
    h5_ids = ids_sorted[inverse]
    X = np.unpackbits(packed, axis=1)[:, :2048].astype(np.float32)

    wanted_positions = {int(index): position for position, index in enumerate(requested)}
    vectors: list[Sequence[int] | None] = [None] * len(requested)
    source_ids: list[object | None] = [None] * len(requested)
    source = pq.ParquetFile(full_plif_path)
    offset = 0
    for batch in source.iter_batches(
        batch_size=batch_size, columns=["ID", "interaction_vector"]
    ):
        stop = offset + batch.num_rows
        left = int(np.searchsorted(sorted_indices, offset, side="left"))
        right = int(np.searchsorted(sorted_indices, stop, side="left"))
        if right > left:
            ids = batch.column("ID").to_pylist()
            interactions = batch.column("interaction_vector").to_pylist()
            for global_index in sorted_indices[left:right]:
                local = int(global_index - offset)
                position = wanted_positions[int(global_index)]
                source_ids[position] = ids[local]
                vectors[position] = interactions[local]
        offset = stop
        if offset > sorted_indices[-1]:
            break
    if any(vector is None for vector in vectors):
        raise RuntimeError("Failed to load all requested interaction vectors")

    decoded_h5_ids = np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in h5_ids],
        dtype=object,
    )
    if not np.array_equal(decoded_h5_ids, np.asarray(source_ids, dtype=object)):
        raise ValueError("HDF5/source PLIF molecule IDs differ for requested rows")

    selected, remaining = interaction_vectors_to_binary(vectors, load_plif_mapping(mapping_path))
    return {
        "molecule_index": requested,
        "X": X,
        "docking": docking,
        "selected": selected,
        "remaining": remaining,
    }
