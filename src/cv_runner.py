"""Outer cross-validation with in-fold scaling and gene selection.

Per outer fold the scaler and every selector see only the training rows; the test rows
are used once, for scoring the fitted classifiers.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src import gwo

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

Selector = Callable[[np.ndarray, np.ndarray, int], np.ndarray]
JSON_COLUMNS = ("per_class_recall", "selected_genes", "history")


def load_or_create_splits(y: np.ndarray, n_splits: int, n_repeats: int, seed: int,
                          path: str | Path, dataset_name: str = "dataset") -> list[dict]:
    """Return frozen outer folds, loading them from ``path`` if it exists.

    The first call generates RepeatedStratifiedKFold folds (``n_splits`` capped by the
    smallest class size) and writes them to ``path``. Later calls load the file and raise
    if it does not match the current data size or CV settings, rather than silently
    regenerating different splits.
    """
    path = Path(path)
    _, counts = np.unique(y, return_counts=True)
    eff_splits = int(min(n_splits, counts.min()))
    if eff_splits < 2:
        raise ValueError("every class needs at least 2 samples for stratified outer CV")
    if eff_splits < n_splits:
        warnings.warn(f"outer n_splits reduced from {n_splits} to {eff_splits} (smallest class)")

    if path.exists():
        saved = json.loads(path.read_text())
        expected = {"n_samples": len(y), "n_splits": eff_splits, "n_repeats": n_repeats, "seed": seed}
        mismatched = {k: (saved.get(k), v) for k, v in expected.items() if saved.get(k) != v}
        if mismatched:
            raise ValueError(f"{path} was saved with different settings {mismatched} (saved, requested); "
                             "delete it or use another out_dir")
        return saved["folds"]

    rskf = RepeatedStratifiedKFold(n_splits=eff_splits, n_repeats=n_repeats, random_state=seed)
    folds = [
        {"repeat": i // eff_splits, "fold": i % eff_splits,
         "train": tr.astype(int).tolist(), "test": te.astype(int).tolist()}
        for i, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), y))
    ]
    payload = {"dataset": dataset_name, "n_samples": len(y), "n_splits": eff_splits,
               "n_repeats": n_repeats, "seed": seed, "folds": folds}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return folds


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "min_recall": float(recall.min()),
            "per_class_recall": recall.tolist(),
        }


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    selectors: Mapping[str, Selector],
    classifiers: Mapping[str, object],
    k: int | Iterable[int] = 30,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    dataset_name: str = "dataset",
    out_dir: str | Path = "results",
    progress: bool = True,
) -> pd.DataFrame:
    """Evaluate every (selector, classifier) pair with frozen repeated stratified CV.

    ``k`` may be a single panel size or several (a k-sweep reuses each fold's scaling, and
    GWO fits are memoised so GWO runs once per fold regardless of the number of k values).
    Writes ``results_<name>.csv`` (list columns as JSON strings), ``splits_<name>.json`` and
    ``convergence_<name>.json`` to ``out_dir`` and returns the results DataFrame.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    ks = [int(k)] if np.isscalar(k) else [int(v) for v in k]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    folds = load_or_create_splits(y, n_splits, n_repeats, seed, out / f"splits_{dataset_name}.json",
                                  dataset_name)
    labels = np.unique(y)

    rows, convergence = [], []
    total = len(folds) * len(selectors) * len(ks)
    bar = tqdm(total=total, desc=dataset_name, disable=not progress) if tqdm else None
    done = 0
    for split in folds:
        tr, te = np.asarray(split["train"]), np.asarray(split["test"])
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        ytr, yte = y[tr], y[te]
        for sel_name, select in selectors.items():
            for kk in ks:
                gwo.reset_last()
                t0 = time.perf_counter()
                genes = np.asarray(select(Xtr, ytr, kk), dtype=int)
                sel_time = time.perf_counter() - t0
                info = gwo.last_info()
                history = None
                if info is not None:
                    history = info["history"]
                    sel_time = info["fit_time_sec"]
                    if not any(c["selector"] == sel_name and c["repeat"] == split["repeat"]
                               and c["fold"] == split["fold"] for c in convergence):
                        convergence.append({"selector": sel_name, "repeat": split["repeat"],
                                            "fold": split["fold"], "history": history})
                for clf_name, clf in classifiers.items():
                    model = clone(clf).fit(Xtr[:, genes], ytr)
                    rows.append({
                        "dataset": dataset_name, "fold": split["fold"], "repeat": split["repeat"],
                        "selector": sel_name, "classifier": clf_name, "k": kk,
                        **_metrics(yte, model.predict(Xte[:, genes]), labels),
                        "n_genes": len(genes), "selected_genes": genes.tolist(),
                        "selection_time_sec": sel_time, "history": history,
                    })
                done += 1
                if bar:
                    bar.update(1)
                elif progress:
                    print(f"[{dataset_name}] {done}/{total} {sel_name} k={kk} "
                          f"(repeat {split['repeat']}, fold {split['fold']})", flush=True)
    if bar:
        bar.close()

    df = pd.DataFrame(rows)
    csv = df.copy()
    for col in JSON_COLUMNS:
        csv[col] = csv[col].map(json.dumps)
    csv.to_csv(out / f"results_{dataset_name}.csv", index=False)
    (out / f"convergence_{dataset_name}.json").write_text(json.dumps(convergence))
    return df


def load_results(dataset_name: str, results_dir: str | Path = "results") -> pd.DataFrame:
    """Read ``results_<name>.csv`` back, decoding the JSON list columns."""
    df = pd.read_csv(Path(results_dir) / f"results_{dataset_name}.csv")
    for col in JSON_COLUMNS:
        df[col] = df[col].map(lambda s: json.loads(s) if isinstance(s, str) else None)
    return df
