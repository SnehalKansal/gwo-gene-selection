"""Figures from saved cv_runner outputs (PNG, dpi 200, written to ``results/figures/``).

Confidence intervals are 95% Student-t intervals across outer folds (all repeats pooled):
mean +/- t_{0.975, n-1} * sd / sqrt(n). Folds from repeated CV share training samples,
so these intervals are optimistic (too narrow); treat them as descriptive.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.cv_runner import load_results

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SELECTOR_ORDER = ["ANOVA", "LASSO", "RF", "mRMR", "GWO", "GWO-chaotic-V", "GWO-free"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def t_ci(values) -> tuple[float, float]:
    """Mean and 95% t-interval half-width of ``values`` (half-width 0 for n < 2)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan), 0.0
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def _colors(names) -> dict[str, str]:
    ordered = [s for s in SELECTOR_ORDER if s in names] + sorted(set(names) - set(SELECTOR_ORDER))
    return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(ordered)}


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig_path(dataset_name: str, stem: str, results_dir: str | Path) -> Path:
    fig_dir = Path(results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir / f"{stem}_{dataset_name}.png"


def plot_convergence(dataset_name: str, results_dir: str | Path = "results",
                     selectors: list[str] | None = None) -> Path:
    """Mean best fitness per iteration (+/- 1 sd band across folds) for each GWO variant."""
    runs = json.loads((Path(results_dir) / f"convergence_{dataset_name}.json").read_text())
    names = sorted({r["selector"] for r in runs})
    if selectors is not None:
        names = [n for n in names if n in selectors]
    colors = _colors(names)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name in names:
        curves = np.array([r["history"] for r in runs if r["selector"] == name], dtype=float)
        mean, sd = curves.mean(axis=0), curves.std(axis=0)
        it = np.arange(1, curves.shape[1] + 1)
        ax.fill_between(it, mean - sd, mean + sd, color=colors[name], alpha=0.18, linewidth=0)
        ax.plot(it, mean, color=colors[name], linewidth=2, label=f"{name} (n={len(curves)} folds)")
    _style(ax)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness (lower is better)")
    ax.set_title(f"GWO convergence, {dataset_name}", color=INK, loc="left")
    ax.legend(frameon=False)
    path = _fig_path(dataset_name, "convergence", results_dir)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_accuracy_vs_genes(dataset_name: str, results_dir: str | Path = "results",
                           metric: str = "accuracy") -> Path:
    """Mean ``metric`` with 95% t-CI vs panel size k; one panel per classifier.

    Requires results at two or more k values. Selectors whose panel size does not follow k
    (free-mode GWO) are omitted.
    """
    df = load_results(dataset_name, results_dir)
    df = df[df["n_genes"] == df["k"]]
    if df["k"].nunique() < 2:
        raise ValueError("accuracy-vs-genes needs runs at two or more k values (use --ks)")
    clfs = sorted(df["classifier"].unique())
    colors = _colors(df["selector"].unique())
    fig, axes = plt.subplots(1, len(clfs), figsize=(4.2 * len(clfs), 3.8), sharey=True, squeeze=False)
    for ax, clf in zip(axes[0], clfs):
        sub = df[df["classifier"] == clf]
        for name, color in colors.items():
            g = sub[sub["selector"] == name]
            if g.empty:
                continue
            ks = sorted(g["k"].unique())
            m, h = zip(*(t_ci(g.loc[g["k"] == kk, metric]) for kk in ks))
            ax.errorbar(ks, m, yerr=h, color=color, linewidth=2, marker="o", markersize=5,
                        capsize=3, label=name)
        _style(ax)
        ax.set_title(clf, color=INK, loc="left")
        ax.set_xlabel("Number of genes (k)")
    axes[0][0].set_ylabel(f"Mean {metric.replace('_', ' ')} (95% CI)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False)
    fig.suptitle(f"{metric.replace('_', ' ').capitalize()} vs panel size, {dataset_name}",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    path = _fig_path(dataset_name, "accuracy_vs_genes", results_dir)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_selector_comparison(dataset_name: str, results_dir: str | Path = "results",
                             k: int | None = None, metric: str = "macro_f1") -> Path:
    """Grouped bars of mean ``metric`` with 95% t-CI per selector (bars) per classifier (groups).

    Uses rows at panel size ``k`` (default: the smallest k present); free-mode GWO rows are
    kept whatever their size.
    """
    df = load_results(dataset_name, results_dir)
    k = int(df["k"].min()) if k is None else k
    df = df[df["k"] == k]
    clfs = sorted(df["classifier"].unique())
    colors = _colors(df["selector"].unique())
    names = list(colors)
    width = 0.8 / len(names)
    x = np.arange(len(clfs))
    fig, ax = plt.subplots(figsize=(max(6.4, 1.6 * len(clfs) * len(names) / 3), 4.2))
    for i, name in enumerate(names):
        stats_ = [t_ci(df[(df["classifier"] == c) & (df["selector"] == name)][metric]) for c in clfs]
        m, h = zip(*stats_)
        ax.bar(x + (i - (len(names) - 1) / 2) * width, m, width * 0.92, yerr=h, color=colors[name],
               capsize=2, error_kw={"ecolor": INK, "elinewidth": 1}, label=name)
    _style(ax)
    ax.set_xticks(x, clfs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"Mean {metric.replace('_', '-')} (95% CI)")
    ax.set_title(f"Selector comparison at k={k}, {dataset_name}", color=INK, loc="left")
    ax.legend(frameon=False, ncol=min(len(names), 4), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    path = _fig_path(dataset_name, "selector_comparison", results_dir)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
