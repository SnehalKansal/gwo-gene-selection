"""Stability of gene selection across outer training folds.

Inputs are the ``selected_genes`` lists in the results file. Each list was produced by a
selector that saw only that fold's training rows, so nothing here involves test data. A
selection is recorded once per (repeat, fold); the duplicate rows for different classifiers
are collapsed.

Indices implemented:
- Kuncheva (2007) consistency index: mean pairwise, corrected for chance; equal-size sets only.
- Jaccard: mean pairwise |A ∩ B| / |A ∪ B|.
- Nogueira et al. (2018) stability: handles variable set sizes, corrected for chance, with an
  asymptotic 95% CI.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SELECTOR_ORDER = ["ANOVA", "LASSO", "RF", "mRMR", "GWO", "GWO-chaotic-V", "GWO-free"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def fold_selections(df: pd.DataFrame, selector: str, k: int | None = None,
                    dataset: str | None = None) -> list[np.ndarray]:
    """One selected-gene array per (repeat, fold) for ``selector`` (optionally one k / dataset)."""
    sub = df[df["selector"] == selector]
    if dataset is not None:
        sub = sub[sub["dataset"] == dataset]
    if k is not None:
        sub = sub[sub["k"] == k]
    if sub["dataset"].nunique() > 1 or sub["k"].nunique() > 1:
        raise ValueError("selections span several datasets or k values; pass dataset= and k=")
    sub = sub.drop_duplicates(["repeat", "fold"]).sort_values(["repeat", "fold"])
    return [np.asarray(g, dtype=int) for g in sub["selected_genes"]]


def selection_matrix(selections: list[np.ndarray], n_features: int) -> np.ndarray:
    """Binary matrix (n_runs, n_features) with 1 where a gene was selected."""
    Z = np.zeros((len(selections), n_features), dtype=bool)
    for i, s in enumerate(selections):
        Z[i, s] = True
    return Z


def kuncheva_index(selections: list[np.ndarray], n_features: int) -> float:
    """Mean pairwise Kuncheva index; all sets must have the same size k (0 < k < p)."""
    sizes = {len(s) for s in selections}
    if len(sizes) != 1:
        raise ValueError("Kuncheva index needs equal-size selections; use nogueira_stability")
    k, p = sizes.pop(), n_features
    if not 0 < k < p:
        raise ValueError("Kuncheva index needs 0 < k < n_features")
    expected = k * k / p
    vals = [(len(np.intersect1d(a, b)) - expected) / (k - expected) for a, b in combinations(selections, 2)]
    return float(np.mean(vals)) if vals else np.nan


def jaccard_index(selections: list[np.ndarray]) -> tuple[float, float]:
    """Mean and SD of pairwise Jaccard similarity."""
    vals = []
    for a, b in combinations(selections, 2):
        union = len(np.union1d(a, b))
        vals.append(len(np.intersect1d(a, b)) / union if union else 1.0)
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, np.nan)


def nogueira_stability(Z: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Nogueira et al. (2018) stability with a (1 - alpha) asymptotic CI.

    ``Z`` is the (M runs × p features) binary selection matrix. Returns ``(phi, lo, hi)``;
    phi = 1 for identical selections, ~0 for random selections of the same sizes.
    """
    Z = np.asarray(Z, dtype=float)
    M, p = Z.shape
    if M < 2:
        return np.nan, np.nan, np.nan
    pf = Z.mean(axis=0)
    kbar = Z.sum(axis=1).mean()
    denom = (kbar / p) * (1 - kbar / p)
    if denom == 0:
        return 1.0, 1.0, 1.0
    s2 = M / (M - 1) * pf * (1 - pf)
    phi = 1 - s2.mean() / denom

    # Asymptotic variance (Nogueira et al. 2018, Theorem 1 / reference implementation).
    ki = Z.sum(axis=1)
    phis = ((Z @ pf) / p - ki * kbar / p ** 2
            - phi / 2 * (2 * kbar * ki / p ** 2 - ki / p - kbar / p + 1)) / denom
    var = 4 / M ** 2 * np.sum((phis - phis.mean()) ** 2)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(phi), float(phi - z * np.sqrt(var)), float(phi + z * np.sqrt(var))


def selection_frequency(selections: list[np.ndarray], genes: np.ndarray | None = None) -> pd.Series:
    """Fraction of runs selecting each gene (only genes selected at least once), sorted desc."""
    flat = np.concatenate(selections) if selections else np.array([], dtype=int)
    idx, counts = np.unique(flat, return_counts=True)
    labels = genes[idx] if genes is not None else idx
    freq = pd.Series(counts / len(selections), index=pd.Index(labels, name="gene"), name="frequency")
    return freq.sort_values(ascending=False, kind="stable")


def stability_table(df: pd.DataFrame, n_features: int | dict[str, int]) -> pd.DataFrame:
    """Kuncheva, Jaccard and Nogueira stability per (dataset, selector, k).

    ``n_features`` is the number of genes in the data the selectors saw: an int, or a dict
    keyed by dataset name (e.g. from ``results/dataset_summary.csv``).
    """
    rows = []
    for (ds, sel, k), _ in df.groupby(["dataset", "selector", "k"], sort=True):
        p = n_features[ds] if isinstance(n_features, dict) else int(n_features)
        sels = fold_selections(df, sel, k=k, dataset=ds)
        sizes = np.array([len(s) for s in sels])
        equal = len(set(sizes)) == 1 and 0 < sizes[0] < p
        phi, lo, hi = nogueira_stability(selection_matrix(sels, p))
        jac, jac_sd = jaccard_index(sels)
        rows.append({
            "dataset": ds, "selector": sel, "k": k, "n_runs": len(sels),
            "mean_size": float(sizes.mean()),
            "kuncheva": kuncheva_index(sels, p) if equal else np.nan,
            "jaccard": jac, "jaccard_sd": jac_sd,
            "nogueira": phi, "nogueira_lo": lo, "nogueira_hi": hi,
        })
    return pd.DataFrame(rows)


def cross_selector_overlap(df: pd.DataFrame, k: int, dataset: str | None = None) -> pd.DataFrame:
    """Mean per-fold Jaccard overlap between every pair of selectors at panel size ``k``."""
    sub = df[df["k"] == k] if dataset is None else df[(df["k"] == k) & (df["dataset"] == dataset)]
    names = [s for s in SELECTOR_ORDER if s in set(sub["selector"])] + sorted(
        set(sub["selector"]) - set(SELECTOR_ORDER))
    per = {n: {(r, f): set(g) for r, f, g in sub[sub["selector"] == n]
               .drop_duplicates(["repeat", "fold"])[["repeat", "fold", "selected_genes"]].itertuples(index=False)}
           for n in names}
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for a in names:
        for b in names:
            common = per[a].keys() & per[b].keys()
            vals = [len(per[a][f] & per[b][f]) / len(per[a][f] | per[b][f]) for f in common
                    if per[a][f] | per[b][f]]
            out.loc[a, b] = np.mean(vals) if vals else np.nan
    return out


def _colors(names) -> dict[str, str]:
    ordered = [s for s in SELECTOR_ORDER if s in names] + sorted(set(names) - set(SELECTOR_ORDER))
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(ordered)}


def plot_stability(table: pd.DataFrame, dataset: str, out_dir: str | Path = "results/figures",
                   metric: str = "nogueira") -> Path:
    """Stability (``metric``) vs k per selector; Nogueira gets its 95% CI as error bars.

    Only fixed-size selectors (mean panel size == k) are drawn; variable-size ones such as
    GWO-free stay in the table but have no meaningful position on a k axis.
    """
    sub = table[(table["dataset"] == dataset) & np.isclose(table["mean_size"], table["k"])]
    colors = _colors(sub["selector"].unique())
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name, color in colors.items():
        g = sub[sub["selector"] == name].sort_values("k")
        if g.empty:
            continue
        yerr = None
        if metric == "nogueira":
            yerr = np.vstack([g["nogueira"] - g["nogueira_lo"], g["nogueira_hi"] - g["nogueira"]])
        ax.errorbar(g["k"], g[metric], yerr=yerr, color=color, linewidth=2, marker="o",
                    markersize=5, capsize=3, label=name)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Number of genes (k)")
    ax.set_ylabel(f"{metric.capitalize()} stability")
    ax.set_title(f"Selection stability across folds, {dataset}", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"stability_{dataset}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
