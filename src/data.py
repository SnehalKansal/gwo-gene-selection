"""Loading and validation of CuMiDa microarray CSV files.

No scaling or label-dependent processing happens here: standardisation and gene selection
are done inside each outer training fold by ``src.cv_runner``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("samples", "type")
RAW_INTENSITY_THRESHOLD = 100.0


class CumidaData(NamedTuple):
    """A loaded CuMiDa dataset plus what was cleaned on load."""

    X: np.ndarray
    y: np.ndarray
    genes: np.ndarray
    class_names: list[str]
    sample_ids: np.ndarray
    n_constant_dropped: int
    n_nan_imputed: int


def load_cumida_with_meta(path: str | Path) -> CumidaData:
    """Load a CuMiDa CSV and return data, labels, gene/probe names, class names and sample ids.

    Validation: the ``samples`` and ``type`` columns must exist, sample ids must be unique and
    there must be at least two classes (``ValueError`` otherwise). Non-numeric expression values
    become NaN; NaNs are imputed with the column median and counted; columns that are constant
    across all samples (after imputation) are dropped and counted. Labels are integer codes of
    the sorted class names (0..C-1). Both cleaning steps are label-free.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required column(s) {missing}")
    sample_ids = df["samples"].astype(str).to_numpy()
    dupes = pd.Series(sample_ids)[pd.Series(sample_ids).duplicated()].unique()
    if len(dupes):
        raise ValueError(f"{path}: duplicate sample ids {list(dupes[:5])}")
    labels = df["type"].astype(str).to_numpy()
    class_names, y = np.unique(labels, return_inverse=True)
    if len(class_names) < 2:
        raise ValueError(f"{path}: need at least 2 classes, found {list(class_names)}")

    expr = df.drop(columns=list(REQUIRED_COLUMNS)).apply(pd.to_numeric, errors="coerce")
    X = expr.to_numpy(dtype=np.float64)
    genes = np.asarray(expr.columns, dtype=str)

    nan_mask = np.isnan(X)
    n_nan = int(nan_mask.sum())
    if n_nan:
        medians = np.nanmedian(np.where(nan_mask.all(axis=0), 0.0, X), axis=0)
        X = np.where(nan_mask, medians, X)
        logger.info("%s: imputed %d NaN values with column medians", path, n_nan)

    constant = np.ptp(X, axis=0) == 0
    n_const = int(constant.sum())
    if n_const:
        X, genes = X[:, ~constant], genes[~constant]
        logger.info("%s: dropped %d constant columns", path, n_const)

    return CumidaData(X, y.astype(np.int64), genes, [str(c) for c in class_names], sample_ids,
                      n_const, n_nan)


def load_cumida(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a CuMiDa CSV as ``(X, y, genes)``; see ``load_cumida_with_meta`` for the checks."""
    d = load_cumida_with_meta(path)
    return d.X, d.y, d.genes


def class_counts(y: np.ndarray, class_names: list[str]) -> dict[str, int]:
    """Number of samples per class name, in class-code order."""
    counts = np.bincount(np.asarray(y), minlength=len(class_names))
    return {name: int(c) for name, c in zip(class_names, counts)}


def log_transform_if_needed(X: np.ndarray) -> tuple[np.ndarray, bool]:
    """Apply ``log2(x - min + 1)`` only if the data look like raw intensities.

    Heuristic: microarray data on a log2 scale rarely exceed ~16, while raw (unlogged)
    intensities reach the thousands, so the transform is applied only when
    ``max(X) > 100``. CuMiDa files are normally already log-scale, so the usual outcome is no
    transform. The shift uses the global minimum (a label-free, sample-independent constant)
    so all values are >= 1 before the log. Returns ``(X_out, transformed)``; the input is not
    modified.
    """
    X = np.asarray(X, dtype=np.float64)
    if np.nanmax(X) <= RAW_INTENSITY_THRESHOLD:
        return X.copy(), False
    return np.log2(X - np.nanmin(X) + 1.0), True
