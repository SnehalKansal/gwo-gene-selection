"""Loading of CuMiDa microarray CSV files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_cumida(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load a CuMiDa CSV (columns: ``samples``, ``type``, then one column per gene).

    Returns ``(X, y, gene_names, class_names)`` where ``X`` is a float array of shape
    (n_samples, n_genes) and ``y`` holds integer class codes indexing ``class_names``.
    """
    df = pd.read_csv(path)
    labels = df["type"].astype(str).to_numpy()
    genes = df.drop(columns=["samples", "type"], errors="ignore")
    class_names, y = np.unique(labels, return_inverse=True)
    X = genes.to_numpy(dtype=float)
    return X, y.astype(int), list(genes.columns), list(class_names)
