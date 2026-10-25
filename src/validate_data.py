"""Week-1 dataset report: class counts and cleaning summary, before any modelling.

    python -m src.validate_data --data data/Leukemia_GSE9476.csv --name Leukemia_GSE9476

Updates (one row per dataset) ``<out>/dataset_summary.csv`` and ``<out>/dataset_summary.md``.
Exits with status 1 and a warning if any class has fewer than ``MIN_CLASS_SIZE`` samples.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import class_counts, load_cumida_with_meta, log_transform_if_needed
from src.tables import to_markdown

MIN_CLASS_SIZE = 5
COLUMNS = ["dataset", "n_samples", "n_features", "n_classes", "class_counts", "imbalance_ratio",
           "min_class_size", "n_constant_features_dropped", "n_nan_imputed", "log_transformed"]


def summarize_dataset(path: str | Path, name: str) -> dict:
    """One summary row for the dataset at ``path``."""
    d = load_cumida_with_meta(path)
    counts = class_counts(d.y, d.class_names)
    _, transformed = log_transform_if_needed(d.X)
    sizes = np.array(list(counts.values()))
    return {
        "dataset": name,
        "n_samples": int(d.X.shape[0]),
        "n_features": int(d.X.shape[1]),
        "n_classes": len(counts),
        "class_counts": "; ".join(f"{k}={v}" for k, v in counts.items()),
        "imbalance_ratio": round(float(sizes.max() / sizes.min()), 2),
        "min_class_size": int(sizes.min()),
        "n_constant_features_dropped": d.n_constant_dropped,
        "n_nan_imputed": d.n_nan_imputed,
        "log_transformed": bool(transformed),
    }


def update_summary(row: dict, out_dir: str | Path = "results") -> pd.DataFrame:
    """Insert or replace ``row`` in the summary CSV and rewrite the markdown table."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "dataset_summary.csv"
    records = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    records = [r for r in records if r["dataset"] != row["dataset"]] + [row]
    table = pd.DataFrame(records, columns=COLUMNS).sort_values("dataset").reset_index(drop=True)
    table.to_csv(csv, index=False)
    (out / "dataset_summary.md").write_text(to_markdown(table) + "\n", encoding="utf-8")
    return table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to a CuMiDa CSV")
    ap.add_argument("--name", help="dataset name (default: file stem)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    name = args.name or Path(args.data).stem
    row = summarize_dataset(args.data, name)
    update_summary(row, args.out)
    for key, value in row.items():
        print(f"{key:>28}: {value}")
    print(f"written: {Path(args.out) / 'dataset_summary.csv'}, {Path(args.out) / 'dataset_summary.md'}")
    if row["min_class_size"] < MIN_CLASS_SIZE:
        print(f"WARNING: {name} has a class with only {row['min_class_size']} samples "
              f"(< {MIN_CLASS_SIZE}). Outer folds will be reduced and GWO's inner CV will be fragile.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
