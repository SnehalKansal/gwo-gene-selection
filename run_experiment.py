"""Run the GWO-vs-baselines gene-selection experiment on one CuMiDa dataset."""
from __future__ import annotations

import argparse
from functools import partial

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src import baselines, gwo, plots
from src.cv_runner import run_cv

QUICK_GWO = {"n_wolves": 10, "n_iter": 15, "prefilter": 1000}


def build_selectors(gwo_params: dict) -> dict:
    """Baselines from ``src.baselines`` (mRMR only if present) plus the GWO variants."""
    selectors = {"ANOVA": baselines.anova_select, "LASSO": baselines.lasso_select,
                 "RF": baselines.rf_select}
    if hasattr(baselines, "mrmr_select"):
        selectors["mRMR"] = baselines.mrmr_select
    gwo_fixed = partial(gwo.gwo_select, **gwo_params)
    selectors["GWO"] = gwo_fixed
    selectors["GWO-chaotic-V"] = partial(gwo.gwo_chaotic_v_select, **gwo_params)
    selectors["GWO-free"] = lambda X, y, k: gwo_fixed(X, y, None)
    return selectors


def build_classifiers() -> dict:
    return {
        "SVM-linear": SVC(kernel="linear", C=1.0),
        "kNN-3": KNeighborsClassifier(n_neighbors=3),
        "RF-200": RandomForestClassifier(n_estimators=200, random_state=0),
    }


def synthetic_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """CuMiDa-shaped stand-in (72 samples, 5000 genes, 3 imbalanced classes) for smoke tests."""
    return make_classification(n_samples=72, n_features=5000, n_informative=10, n_redundant=10,
                               n_classes=3, weights=[0.5, 0.3, 0.2], class_sep=2.0,
                               random_state=seed)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["k", "selector", "classifier"] if df["k"].nunique() > 1 else ["selector", "classifier"]
    return (df.groupby(keys)[["accuracy", "macro_f1", "min_recall", "n_genes"]]
              .mean().round(3))


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="path to a CuMiDa CSV")
    ap.add_argument("--name", help="dataset name used in output files")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data instead of --data")
    ap.add_argument("--k", type=int, default=30, help="panel size for the main comparison")
    ap.add_argument("--ks", type=int, nargs="+", help="panel sizes for an accuracy-vs-genes sweep")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="n_wolves=10, n_iter=15, prefilter=1000")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    if args.synthetic:
        X, y = synthetic_data()
        name = args.name or "synthetic"
    elif args.data:
        from src.data import load_cumida
        X, y, *_ = load_cumida(args.data)
        name = args.name or args.data.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".csv")
    else:
        ap.error("give --data PATH or --synthetic")
    print(f"{name}: {X.shape[0]} samples, {X.shape[1]} genes, classes {np.unique(y, return_counts=True)}")

    ks = sorted(set(args.ks or [])) or [args.k]
    if args.k not in ks:
        ks = sorted(set(ks) | {args.k})
    df = run_cv(X, y, build_selectors(QUICK_GWO if args.quick else {}), build_classifiers(), k=ks,
                n_splits=args.splits, n_repeats=args.repeats, seed=42, dataset_name=name,
                out_dir=args.out)

    figs = [plots.plot_convergence(name, args.out, selectors=["GWO", "GWO-chaotic-V"]),
            plots.plot_selector_comparison(name, args.out, k=args.k)]
    if len(ks) > 1:
        figs.append(plots.plot_accuracy_vs_genes(name, args.out))
    with pd.option_context("display.width", 140, "display.max_rows", 200):
        print(summarize(df))
    print("figures:", *figs, sep="\n  ")
    return df


if __name__ == "__main__":
    main()
