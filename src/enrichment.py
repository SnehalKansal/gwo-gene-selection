"""Exploratory GO / pathway over-representation analysis of selected gene panels.

Only genes chosen inside outer training folds are used: a panel is the set of genes selected
in at least ``min_freq`` of the folds (``consensus_genes``). CuMiDa columns are usually
Affymetrix probe IDs, so probes are mapped to gene symbols with a platform annotation file
(e.g. the GEO GPL table) before testing. The background is every gene measured on the array
(after mapping), not the whole genome.

The test is a one-sided hypergeometric test per gene set with Benjamini-Hochberg FDR.
Gene sets come from a local GMT file (e.g. GO Biological Process from MSigDB or Enrichr's
library downloads), so the analysis runs offline. ``run_enrichr`` optionally queries Enrichr
through gseapy if it is installed (network access needed). Results are hypothesis-generating
only.
"""
from __future__ import annotations

import gzip
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AFFY_PROBE = re.compile(r"^(AFFX-)?[\w.-]+_(s_|x_|a_|i_|f_|g_|r_)?at$", re.IGNORECASE)


def looks_like_probe_ids(names, threshold: float = 0.5) -> bool:
    """True if at least ``threshold`` of ``names`` look like Affymetrix probe IDs (``..._at``)."""
    names = [str(n) for n in names]
    return bool(names) and np.mean([bool(AFFY_PROBE.match(n)) for n in names]) >= threshold


def load_gmt(path: str | Path) -> dict[str, set[str]]:
    """Read a GMT file (name<TAB>description<TAB>gene1<TAB>gene2...) into {name: set of symbols}."""
    sets = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            sets[parts[0]] = {g.strip().upper() for g in parts[2:] if g.strip()}
    return sets


SYMBOL_COLUMNS = ("gene symbol", "symbol", "gene_symbol", "gene.symbol")


def load_probe_annotation(path: str | Path, probe_col: str = "ID",
                          symbol_col: str | None = None) -> dict[str, list[str]]:
    """Probe -> list of gene symbols from a platform annotation table.

    Accepts GEO platform files (``GPLxxx.annot(.gz)`` or GPL full tables: tab-separated, with
    ``!``/``^``/``#`` preamble lines) and plain CSV/TSV tables; ``.gz`` is read transparently.
    The header is the first line that starts with ``probe_col``. ``symbol_col`` defaults to the
    first column named like "Gene symbol" (case-insensitive). Multi-gene entries separated by
    ``///`` (Affymetrix convention) are split; probes without a symbol are left out.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh if ln.strip() and not ln.startswith(("!", "^", "#"))]
    start = next((i for i, ln in enumerate(lines) if ln.split("\t")[0].split(",")[0].strip('"') == probe_col), None)
    if start is None:
        raise ValueError(f"{path}: no header line starting with {probe_col!r}")
    sep = "\t" if "\t" in lines[start] else ","
    ann = pd.read_csv(io.StringIO("\n".join(lines[start:])), sep=sep, dtype=str)
    if symbol_col is None:
        symbol_col = next((c for c in ann.columns if c.strip().lower() in SYMBOL_COLUMNS), None)
        if symbol_col is None:
            raise ValueError(f"{path}: no gene-symbol column among {list(ann.columns)}")
    mapping = {}
    for probe, sym in zip(ann[probe_col], ann[symbol_col]):
        if isinstance(sym, str) and sym.strip() and sym.strip() != "---":
            mapping[str(probe)] = [s.strip().upper() for s in sym.split("///") if s.strip()]
    return mapping


def map_to_symbols(names, mapping: dict[str, list[str]] | None = None) -> tuple[set[str], int]:
    """Map probe IDs (or symbols) to a set of upper-case symbols; returns (symbols, n_unmapped).

    With ``mapping=None`` the names are taken to be symbols already.
    """
    names = [str(n) for n in names]
    if mapping is None:
        return {n.upper() for n in names}, 0
    symbols, unmapped = set(), 0
    for n in names:
        hit = mapping.get(n)
        if hit:
            symbols.update(hit)
        else:
            unmapped += 1
    return symbols, unmapped


def consensus_genes(df: pd.DataFrame, selector: str, genes: np.ndarray, k: int | None = None,
                    dataset: str | None = None, min_freq: float = 0.5) -> pd.Series:
    """Genes selected in at least ``min_freq`` of the outer training folds, with their frequency."""
    from src.stability import fold_selections, selection_frequency

    freq = selection_frequency(fold_selections(df, selector, k=k, dataset=dataset), np.asarray(genes))
    return freq[freq >= min_freq]


def benjamini_hochberg(p) -> np.ndarray:
    """BH-adjusted p-values (q-values), same order as the input."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def ora(selected: set[str], background: set[str], gene_sets: dict[str, set[str]],
        min_size: int = 5, max_size: int = 500) -> pd.DataFrame:
    """Hypergeometric over-representation of ``selected`` in each gene set, within ``background``.

    Gene sets are intersected with the background and kept if their size is in
    [min_size, max_size]. Returns one row per tested set, sorted by p-value, with the overlap,
    fold enrichment, p-value and BH q-value.
    """
    background = {g.upper() for g in background}
    selected = {g.upper() for g in selected} & background
    N, n = len(background), len(selected)
    rows = []
    for name, members in gene_sets.items():
        members = members & background
        K = len(members)
        if not min_size <= K <= max_size:
            continue
        overlap = selected & members
        x = len(overlap)
        p = float(stats.hypergeom.sf(x - 1, N, K, n)) if x > 0 else 1.0
        rows.append({"gene_set": name, "set_size": K, "overlap": x,
                     "expected": K * n / N if N else np.nan,
                     "fold_enrichment": (x / (K * n / N)) if N and K and n else np.nan,
                     "p_value": p, "genes": ",".join(sorted(overlap))})
    out = pd.DataFrame(rows, columns=["gene_set", "set_size", "overlap", "expected",
                                      "fold_enrichment", "p_value", "genes"])
    out["q_value"] = benjamini_hochberg(out["p_value"].to_numpy())
    return out.sort_values(["p_value", "gene_set"]).reset_index(drop=True)


def run_enrichr(symbols, gene_sets: str = "GO_Biological_Process_2023",
                background: list[str] | None = None) -> pd.DataFrame:
    """Optional: Enrichr query via gseapy (needs ``pip install gseapy`` and network access)."""
    try:
        import gseapy
    except ImportError as exc:
        raise ImportError("run_enrichr needs gseapy; use ora() with a local GMT file instead") from exc
    res = gseapy.enrichr(gene_list=sorted(symbols), gene_sets=gene_sets, background=background,
                         outdir=None)
    return res.results
