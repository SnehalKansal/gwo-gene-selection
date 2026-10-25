"""Results tables: performance summaries, paired tests and markdown export.

Reads ``results/results_<dataset>.csv`` as written by ``src.cv_runner`` (one row per
repeat × fold × selector × classifier [× k]). Confidence intervals are 95% Student-t intervals
across outer folds; folds of repeated CV share training samples, so intervals and p-values are
optimistic and reported as descriptive.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

JSON_COLUMNS = ("per_class_recall", "selected_genes", "history")
METRICS = ("accuracy", "macro_f1", "balanced_accuracy", "min_recall")


def read_results(path: str | Path) -> pd.DataFrame:
    """Load a results CSV, decode JSON list columns, and add ``k`` (= n_genes) if absent."""
    df = pd.read_csv(path)
    for col in JSON_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(lambda s: json.loads(s) if isinstance(s, str) else None)
    if "k" not in df.columns:
        df["k"] = df["n_genes"]
    return df


def results_path(dataset_name: str, results_dir: str | Path = "results") -> Path:
    return Path(results_dir) / f"results_{dataset_name}.csv"


def mean_ci(values) -> tuple[float, float, float]:
    """Mean and 95% t-interval bounds (bounds equal the mean when n < 2)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    m = float(v.mean())
    if len(v) < 2:
        return m, m, m
    h = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return m, m - h, m + h


def performance_table(df: pd.DataFrame, k: int | None = None,
                      metrics: tuple[str, ...] = ("accuracy", "macro_f1", "min_recall")) -> pd.DataFrame:
    """Mean and 95% CI per (dataset, k, selector, classifier) for each metric.

    Columns per metric: ``<m>_mean``, ``<m>_lo``, ``<m>_hi``, plus ``n_folds`` and mean
    ``n_genes``. Restricted to rows with ``k`` if given.
    """
    if k is not None:
        df = df[df["k"] == k]
    rows = []
    for keys, g in df.groupby(["dataset", "k", "selector", "classifier"], sort=True):
        row = dict(zip(["dataset", "k", "selector", "classifier"], keys))
        for m in metrics:
            row[f"{m}_mean"], row[f"{m}_lo"], row[f"{m}_hi"] = mean_ci(g[m])
        row["n_genes"] = float(g["n_genes"].mean())
        row["n_folds"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def format_table(perf: pd.DataFrame, metrics: tuple[str, ...] = ("accuracy", "macro_f1", "min_recall"),
                 digits: int = 3) -> pd.DataFrame:
    """Human-readable version of ``performance_table``: ``mean [lo, hi]`` strings."""
    out = perf[["dataset", "k", "selector", "classifier"]].copy()
    for m in metrics:
        out[m] = [f"{a:.{digits}f} [{b:.{digits}f}, {c:.{digits}f}]"
                  for a, b, c in zip(perf[f"{m}_mean"], perf[f"{m}_lo"], perf[f"{m}_hi"])]
    out["n_genes"] = perf["n_genes"].round(1)
    return out


def paired_tests(df: pd.DataFrame, reference: str = "GWO", metric: str = "macro_f1") -> pd.DataFrame:
    """Two-sided Wilcoxon signed-rank test of ``reference`` vs each other selector.

    Pairs are matched on (dataset, k, classifier, repeat, fold). Holm correction is applied
    within each (dataset, k, classifier). ``mean_diff`` is reference minus other. Folds are not
    independent, so p-values are descriptive.
    """
    keys = ["dataset", "k", "classifier", "repeat", "fold"]
    wide = df.pivot_table(index=keys, columns="selector", values=metric, aggfunc="first")
    if reference not in wide.columns:
        raise ValueError(f"reference selector {reference!r} not in results")
    rows = []
    for (ds, k, clf), g in wide.groupby(level=["dataset", "k", "classifier"]):
        for other in [c for c in wide.columns if c != reference]:
            pair = g[[reference, other]].dropna()
            diff = pair[reference] - pair[other]
            if len(pair) < 2 or np.allclose(diff, 0):
                p = 1.0
            else:
                p = float(stats.wilcoxon(pair[reference], pair[other]).pvalue)
            rows.append({"dataset": ds, "k": k, "classifier": clf, "reference": reference,
                         "other": other, "n_pairs": len(pair), "mean_diff": float(diff.mean()),
                         "p_value": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = out.groupby(["dataset", "k", "classifier"])["p_value"].transform(holm)
    return out


def holm(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out


def friedman_ranks(df: pd.DataFrame, metric: str = "macro_f1") -> tuple[pd.Series, float, float]:
    """Mean rank of each selector (1 = best) over (dataset, k, classifier, repeat, fold) blocks,
    plus the Friedman chi-square statistic and p-value. Blocks with missing selectors are dropped.
    """
    keys = ["dataset", "k", "classifier", "repeat", "fold"]
    wide = df.pivot_table(index=keys, columns="selector", values=metric, aggfunc="first").dropna()
    ranks = wide.rank(axis=1, ascending=False).mean().sort_values()
    if wide.shape[1] < 3 or len(wide) < 2:
        return ranks, np.nan, np.nan
    stat, p = stats.friedmanchisquare(*[wide[c] for c in wide.columns])
    return ranks, float(stat), float(p)


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """GitHub-flavoured markdown table (no ``tabulate`` dependency)."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v).replace("|", "\\|")

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def write_tables(dataset_name: str, results_dir: str | Path = "results", k: int | None = None,
                 reference: str = "GWO") -> dict[str, Path]:
    """Write performance, paired-test and rank tables (CSV + markdown) to ``<results_dir>/tables``."""
    df = read_results(results_path(dataset_name, results_dir))
    out = Path(results_dir) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    written = {}

    perf = performance_table(df, k)
    perf.to_csv(out / f"performance_{dataset_name}.csv", index=False)
    md = to_markdown(format_table(perf))
    (out / f"performance_{dataset_name}.md").write_text(md + "\n", encoding="utf-8")
    written["performance"] = out / f"performance_{dataset_name}.md"

    sub = df if k is None else df[df["k"] == k]
    if reference in set(sub["selector"]):
        tests = paired_tests(sub, reference)
        tests.to_csv(out / f"paired_tests_{dataset_name}.csv", index=False)
        (out / f"paired_tests_{dataset_name}.md").write_text(to_markdown(tests) + "\n", encoding="utf-8")
        written["paired_tests"] = out / f"paired_tests_{dataset_name}.md"

    fixed = sub[sub["n_genes"] == sub["k"]]
    ranks, stat, p = friedman_ranks(fixed)
    rank_df = ranks.rename("mean_rank").reset_index()
    text = to_markdown(rank_df) + f"\n\nFriedman chi2 = {stat:.3f}, p = {p:.3g} (fixed-k selectors only)\n"
    (out / f"ranks_{dataset_name}.md").write_text(text, encoding="utf-8")
    written["ranks"] = out / f"ranks_{dataset_name}.md"
    return written


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Write performance, paired-test and rank tables.")
    ap.add_argument("--name", required=True, help="dataset name (results_<name>.csv)")
    ap.add_argument("--results", default="results")
    ap.add_argument("--k", type=int, help="panel size to tabulate (default: all)")
    ap.add_argument("--reference", default="GWO", help="selector compared against the others")
    args = ap.parse_args(argv)
    for kind, path in write_tables(args.name, args.results, args.k, args.reference).items():
        print(f"{kind:>13}: {path}")


if __name__ == "__main__":
    main()
