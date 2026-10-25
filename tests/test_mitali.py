"""Tests for Mitali's modules: data, baselines, stability, enrichment, tables, validate_data."""
from __future__ import annotations

import json
from functools import partial
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.datasets import make_classification

from src import baselines, enrichment, stability, tables, validate_data
from src.classifiers import get_classifiers
from src.data import class_counts, load_cumida, load_cumida_with_meta, log_transform_if_needed

SELECTORS = {
    "ANOVA": baselines.anova_select,
    "LASSO": baselines.lasso_select,
    "LASSO-C1": partial(baselines.lasso_select, C=1.0),
    "LASSO-multinomial": partial(baselines.lasso_select, C=1.0, multiclass="multinomial"),
    "RF": partial(baselines.rf_select, n_estimators=100),
    "mRMR": baselines.mrmr_select,
    "mRMR-mid": partial(baselines.mrmr_select, scheme="mid"),
}
INFORMATIVE = set(range(5))


def make_data(n_classes=2, n=60, p=300, seed=0):
    # shuffle=False puts the informative genes in columns 0-4.
    return make_classification(n_samples=n, n_features=p, n_informative=5, n_redundant=0,
                               n_repeated=0, n_classes=n_classes, n_clusters_per_class=1,
                               shuffle=False, class_sep=2.0, random_state=seed)


def write_cumida(path, X, labels, ids=None, genes=None):
    genes = genes if genes is not None else [f"{i}_at" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=genes)
    df.insert(0, "type", labels)
    df.insert(0, "samples", ids if ids is not None else [f"GSM{i}" for i in range(len(X))])
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- data

def test_load_cumida_cleans_and_encodes(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(5, 1, size=(12, 6))
    X[:, 2] = 3.0
    X[1, 0] = X[4, 5] = np.nan
    labels = ["tumor", "normal", "tumor"] * 4
    path = write_cumida(tmp_path / "d.csv", X, labels)
    d = load_cumida_with_meta(path)
    assert d.n_constant_dropped == 1 and d.n_nan_imputed == 2
    assert d.X.shape == (12, 5) and d.X.dtype == np.float64 and not np.isnan(d.X).any()
    assert "2_at" not in d.genes
    assert d.class_names == ["normal", "tumor"]
    assert list(d.y[:3]) == [1, 0, 1]
    assert d.X[1, 0] == pytest.approx(np.nanmedian(X[:, 0]))
    assert class_counts(d.y, d.class_names) == {"normal": 4, "tumor": 8}
    X3, y3, g3 = load_cumida(path)
    assert np.array_equal(X3, d.X) and np.array_equal(g3, d.genes)


@pytest.mark.parametrize("problem", ["missing", "dupes", "one_class"])
def test_load_cumida_rejects_bad_files(tmp_path, problem):
    X = np.random.default_rng(0).normal(size=(6, 3))
    labels, ids = ["a", "b"] * 3, None
    if problem == "dupes":
        ids = ["s1", "s1", "s2", "s3", "s4", "s5"]
    if problem == "one_class":
        labels = ["a"] * 6
    path = write_cumida(tmp_path / "bad.csv", X, labels, ids)
    if problem == "missing":
        pd.read_csv(path).drop(columns="type").to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_cumida(path)


def test_log_transform_heuristic():
    logged = np.random.default_rng(0).uniform(2, 14, size=(5, 4))
    out, flag = log_transform_if_needed(logged)
    assert not flag and np.array_equal(out, logged)
    raw = 2.0 ** logged
    before = raw.copy()
    out, flag = log_transform_if_needed(raw)
    assert flag and out.max() < 20 and out.min() == pytest.approx(0.0)
    assert np.array_equal(raw, before)


# ---------------------------------------------------------------- baselines

@pytest.mark.parametrize("name", SELECTORS)
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("k", [1, 10, 40])
def test_selector_interface(name, n_classes, k):
    X, y = make_data(n_classes)
    idx = SELECTORS[name](X, y, k)
    assert isinstance(idx, np.ndarray) and np.issubdtype(idx.dtype, np.integer)
    assert len(idx) == k == len(np.unique(idx))
    assert idx.min() >= 0 and idx.max() < X.shape[1]


@pytest.mark.parametrize("name", SELECTORS)
def test_selector_handles_k_beyond_prefilter_and_candidates(name):
    X, y = make_data(p=40)
    idx = SELECTORS[name](X, y, 40)
    assert sorted(idx) == list(range(40))


@pytest.mark.parametrize("name", SELECTORS)
def test_selector_rejects_invalid_k(name):
    X, y = make_data(p=30)
    for k in (0, 31):
        with pytest.raises(ValueError):
            SELECTORS[name](X, y, k)


@pytest.mark.parametrize("name", SELECTORS)
def test_selector_does_not_modify_inputs(name):
    X, y = make_data(3)
    X0, y0 = X.copy(), y.copy()
    SELECTORS[name](X, y, 10)
    assert np.array_equal(X, X0) and np.array_equal(y, y0)


@pytest.mark.parametrize("name", SELECTORS)
def test_selector_deterministic_and_stateless(name):
    """Same input -> same output, unaffected by calls on other data in between (no caches)."""
    X, y = make_data(seed=0)
    X2, y2 = make_data(seed=1)
    first = SELECTORS[name](X, y, 10)
    SELECTORS[name](X2, y2, 10)
    assert np.array_equal(first, SELECTORS[name](X.copy(), y.copy(), 10))


@pytest.mark.parametrize("name", SELECTORS)
def test_selector_uses_only_rows_passed(name):
    """Changing rows that are not passed in cannot change the result."""
    X, y = make_data(n=80)
    train = np.arange(60)
    a = SELECTORS[name](X[train], y[train], 10)
    X_mod = X.copy()
    X_mod[60:] = np.random.default_rng(1).normal(size=X_mod[60:].shape) * 100
    b = SELECTORS[name](X_mod[train], y[train], 10)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", SELECTORS)
def test_selector_finds_informative_genes(name):
    X, y = make_data()
    assert len(INFORMATIVE & set(SELECTORS[name](X, y, 10).tolist())) >= 3


def test_seed_changes_randomised_selectors_only_via_argument():
    X, y = make_data(3)
    a = baselines.rf_select(X, y, 10, n_estimators=50, seed=0)
    b = baselines.rf_select(X, y, 10, n_estimators=50, seed=0)
    assert np.array_equal(a, b)


def test_mrmr_avoids_redundant_copies():
    """With 4 near-copies of the top gene, ANOVA takes all 5; mRMR's second pick is never a copy
    and it takes fewer copies overall. (With *mean* redundancy, later copies can return once
    the panel grows, because one very relevant gene outweighs a diluted redundancy term.)"""
    X, y = make_data()
    rng = np.random.default_rng(0)
    top = baselines.anova_select(X, y, 1)[0]
    copies = X[:, [top] * 4] + rng.normal(scale=1e-3, size=(len(X), 4))
    Xc = np.hstack([X, copies])
    copy_ids = {top, *range(X.shape[1], X.shape[1] + 4)}
    assert len(copy_ids & set(baselines.anova_select(Xc, y, 5).tolist())) == 5
    for scheme in ("miq", "mid"):
        sel = baselines.mrmr_select(Xc, y, 5, scheme=scheme).tolist()
        assert sel[0] in copy_ids and sel[1] not in copy_ids
        assert len(copy_ids & set(sel)) < 5


def test_lasso_fill_order_when_few_nonzero():
    X, y = make_data()
    idx = baselines.lasso_select(X, y, 50, C=0.01)
    assert len(idx) == 50 == len(set(idx))
    F = baselines._f_scores(X, y)
    tail = idx[-20:]
    assert np.all(np.diff(F[tail]) <= 1e-9)


def test_prefilter_nan_safe_and_capped():
    X, y = make_data(p=50)
    X = X.copy()
    X[:, 3] = 1.0
    idx, F = baselines._prefilter(X, y, n=2000)
    assert len(idx) == 50 and np.isfinite(F).all() and F[3] == 0


def test_classifiers_match_runner_names():
    assert set(get_classifiers()) == {"SVM-linear", "kNN-3", "RF-200"}


def test_runs_inside_snehals_cv_runner(tmp_path):
    """Integration: the selectors plug into src.cv_runner.run_cv unchanged."""
    from src.cv_runner import run_cv

    X, y = make_data(3, n=45, p=120)
    sel = {"ANOVA": baselines.anova_select, "mRMR": baselines.mrmr_select,
           "LASSO": partial(baselines.lasso_select, C=1.0)}
    df = run_cv(X, y, sel, {"kNN-3": get_classifiers()["kNN-3"]}, k=8, n_splits=3, n_repeats=1,
                dataset_name="toy", out_dir=tmp_path, progress=False)
    assert set(df["selector"]) == set(sel) and (df["n_genes"] == 8).all()
    loaded = tables.read_results(tmp_path / "results_toy.csv")
    assert loaded["selected_genes"].map(len).eq(8).all()


# ---------------------------------------------------------------- synthetic results file

@pytest.fixture
def results_csv(tmp_path):
    """Synthetic results in cv_runner's format (no k/history columns, like the shared spec)."""
    rng = np.random.default_rng(0)
    p, rows = 200, []
    stable = np.arange(10)
    for repeat in range(2):
        for fold in range(5):
            picks = {
                "Stable": stable,
                "Random": rng.choice(p, 10, replace=False),
                "GWO": np.r_[stable[:5], rng.choice(np.arange(10, p), 5, replace=False)],
            }
            for sel, genes in picks.items():
                for clf, base in (("kNN-3", 0.7), ("SVM-linear", 0.8)):
                    acc = base + (0.1 if sel == "Stable" else 0) + rng.normal(0, 0.02)
                    rows.append({"dataset": "toy", "fold": fold, "repeat": repeat, "selector": sel,
                                 "classifier": clf, "accuracy": acc, "macro_f1": acc - 0.05,
                                 "balanced_accuracy": acc, "min_recall": acc - 0.1,
                                 "per_class_recall": json.dumps([acc, acc - 0.1]), "n_genes": 10,
                                 "selected_genes": json.dumps(genes.tolist()),
                                 "selection_time_sec": 0.1})
    path = tmp_path / "results_toy.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path, p


# ---------------------------------------------------------------- stability

def test_fold_selections_dedupes_classifier_rows(results_csv):
    path, _ = results_csv
    df = tables.read_results(path)
    sels = stability.fold_selections(df, "Stable")
    assert len(sels) == 10
    assert all(np.array_equal(s, np.arange(10)) for s in sels)


def test_stability_indices_known_values():
    p = 100
    same = [np.arange(10)] * 5
    assert stability.kuncheva_index(same, p) == pytest.approx(1.0)
    assert stability.jaccard_index(same)[0] == pytest.approx(1.0)
    assert stability.nogueira_stability(stability.selection_matrix(same, p))[0] == pytest.approx(1.0)

    a, b = np.arange(10), np.arange(5, 15)
    assert stability.jaccard_index([a, b])[0] == pytest.approx(5 / 15)
    assert stability.kuncheva_index([a, b], p) == pytest.approx((5 - 1) / (10 - 1))

    rng = np.random.default_rng(0)
    rand = [rng.choice(p, 10, replace=False) for _ in range(200)]
    assert abs(stability.kuncheva_index(rand, p)) < 0.05
    phi, lo, hi = stability.nogueira_stability(stability.selection_matrix(rand, p))
    assert abs(phi) < 0.05 and lo < phi < hi


def test_nogueira_matches_kuncheva_for_equal_sizes():
    """For equal-size sets Nogueira's phi equals the mean pairwise Kuncheva index."""
    rng = np.random.default_rng(1)
    p = 60
    sels = [np.r_[np.arange(5), rng.choice(np.arange(5, p), 5, replace=False)] for _ in range(8)]
    phi = stability.nogueira_stability(stability.selection_matrix(sels, p))[0]
    assert phi == pytest.approx(stability.kuncheva_index(sels, p), abs=1e-9)


def test_kuncheva_requires_equal_sizes():
    with pytest.raises(ValueError):
        stability.kuncheva_index([np.arange(5), np.arange(6)], 50)


def test_stability_table_orders_selectors(results_csv, tmp_path):
    path, p = results_csv
    table = stability.stability_table(tables.read_results(path), p).set_index("selector")
    assert table.loc["Stable", "nogueira"] == pytest.approx(1.0)
    assert table.loc["Stable", "nogueira"] > table.loc["GWO", "nogueira"] > table.loc["Random", "nogueira"]
    assert (table["n_runs"] == 10).all()
    fig = stability.plot_stability(table.reset_index(), "toy", tmp_path)
    assert fig.exists()


def test_selection_frequency_and_overlap(results_csv):
    path, p = results_csv
    df = tables.read_results(path)
    genes = np.array([f"g{i}" for i in range(p)])
    freq = stability.selection_frequency(stability.fold_selections(df, "GWO"), genes)
    assert (freq.loc[[f"g{i}" for i in range(5)]] == 1.0).all()
    assert freq.is_monotonic_decreasing
    ov = stability.cross_selector_overlap(df, k=10)
    assert ov.loc["Stable", "Stable"] == 1.0
    assert ov.loc["GWO", "Stable"] == pytest.approx(ov.loc["Stable", "GWO"])


# ---------------------------------------------------------------- enrichment

def test_gmt_and_annotation_parsing(tmp_path):
    gmt = tmp_path / "sets.gmt"
    gmt.write_text("SET_A\tdesc\tTP53\tmyc\tEGFR\nSET_B\thttp://x\tBRCA1\n", encoding="utf-8")
    sets = enrichment.load_gmt(gmt)
    assert sets == {"SET_A": {"TP53", "MYC", "EGFR"}, "SET_B": {"BRCA1"}}

    ann = tmp_path / "gpl.csv"
    ann.write_text("#comment line\nID,Gene Symbol\n1007_s_at,DDR1 /// MIR4640\n1053_at,RFC2\n"
                   "117_at,---\n121_at,\n", encoding="utf-8")
    mapping = enrichment.load_probe_annotation(ann)
    assert mapping == {"1007_s_at": ["DDR1", "MIR4640"], "1053_at": ["RFC2"]}
    syms, unmapped = enrichment.map_to_symbols(["1007_s_at", "117_at", "1053_at"], mapping)
    assert syms == {"DDR1", "MIR4640", "RFC2"} and unmapped == 1
    assert enrichment.looks_like_probe_ids(["1007_s_at", "1053_at", "AFFX-BioB-5_at"])
    assert not enrichment.looks_like_probe_ids(["TP53", "MYC"])


def test_ora_hypergeometric_and_fdr():
    background = {f"G{i}" for i in range(100)}
    sets = {"HIT": {f"G{i}" for i in range(10)}, "MISS": {f"G{i}" for i in range(50, 60)},
            "TINY": {"G1", "G2"}, "OUTSIDE": {f"X{i}" for i in range(20)}}
    selected = {f"G{i}" for i in range(5)} | {"G70", "NOT_ON_ARRAY"}
    res = enrichment.ora(selected, background, sets).set_index("gene_set")
    assert set(res.index) == {"HIT", "MISS"}
    assert res.loc["HIT", "overlap"] == 5
    assert res.loc["HIT", "p_value"] == pytest.approx(stats.hypergeom.sf(4, 100, 10, 6))
    assert res.loc["MISS", "p_value"] == 1.0
    assert res.loc["HIT", "q_value"] >= res.loc["HIT", "p_value"]


def test_benjamini_hochberg_reference():
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.5])
    expected = np.array([0.025, 0.05, 0.05, 0.025, 0.5])
    np.testing.assert_allclose(enrichment.benjamini_hochberg(p), expected)


def test_consensus_genes_uses_fold_frequency(results_csv):
    path, p = results_csv
    df = tables.read_results(path)
    genes = np.array([f"g{i}" for i in range(p)])
    cons = enrichment.consensus_genes(df, "GWO", genes, min_freq=0.9)
    assert set(cons.index) == {f"g{i}" for i in range(5)}


# ---------------------------------------------------------------- tables

def test_performance_and_paired_tables(results_csv, tmp_path):
    path, _ = results_csv
    df = tables.read_results(path)
    assert (df["k"] == 10).all()
    perf = tables.performance_table(df).set_index(["selector", "classifier"])
    row = perf.loc[("Stable", "kNN-3")]
    assert row["n_folds"] == 10 and row["accuracy_lo"] < row["accuracy_mean"] < row["accuracy_hi"]
    tests = tables.paired_tests(df, reference="GWO").set_index(["classifier", "other"])
    assert tests.loc[("kNN-3", "Stable"), "mean_diff"] < 0
    assert tests.loc[("kNN-3", "Stable"), "p_holm"] < 0.05
    ranks, stat, pval = tables.friedman_ranks(df)
    assert ranks.index[0] == "Stable" and pval < 0.05

    written = tables.write_tables("toy", tmp_path, reference="GWO")
    md = written["performance"].read_text()
    assert md.startswith("| dataset | k | selector | classifier | accuracy |")


def test_holm_reference():
    np.testing.assert_allclose(tables.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_to_markdown_escapes_and_formats():
    md = tables.to_markdown(pd.DataFrame({"a": ["x|y"], "b": [0.12345]}))
    assert md.splitlines() == ["| a | b |", "|---|---|", "| x\\|y | 0.123 |"]


# ---------------------------------------------------------------- validate_data

def test_validate_data_cli_updates_rows(tmp_path, capsys):
    X = np.random.default_rng(0).normal(size=(40, 20))
    labels = ["normal", "tumor"] * 20
    data = write_cumida(tmp_path / "a.csv", X, labels)
    out = tmp_path / "res"
    assert validate_data.main(["--data", str(data), "--name", "A", "--out", str(out)]) == 0
    assert validate_data.main(["--data", str(data), "--name", "A", "--out", str(out)]) == 0
    summary = pd.read_csv(out / "dataset_summary.csv")
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_samples"] == 40 and row["n_features"] == 20 and row["n_classes"] == 2
    assert row["class_counts"] == "normal=20; tumor=20" and row["imbalance_ratio"] == 1.0
    assert "| A |" in (out / "dataset_summary.md").read_text()


def test_validate_data_warns_on_tiny_class(tmp_path, capsys):
    X = np.random.default_rng(0).normal(size=(13, 5))
    labels = ["big"] * 10 + ["small"] * 3
    data = write_cumida(tmp_path / "t.csv", X, labels)
    assert validate_data.main(["--data", str(data), "--name", "T", "--out", str(tmp_path)]) == 1
    assert "WARNING" in capsys.readouterr().err
