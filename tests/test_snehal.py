"""Tests for src/gwo.py and src/cv_runner.py (Snehal)."""
from __future__ import annotations

import json
from functools import partial

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src import cv_runner, gwo
from src.cv_runner import run_cv

QUICK = {"n_wolves": 10, "n_iter": 15, "prefilter": 100}
INFORMATIVE = set(range(5))


@pytest.fixture(autouse=True)
def fresh_cache():
    gwo.clear_cache()
    gwo.reset_last()
    yield
    gwo.clear_cache()


@pytest.fixture
def data():
    # shuffle=False puts the 5 informative genes in columns 0-4.
    return make_classification(n_samples=60, n_features=300, n_informative=5, n_redundant=0,
                               n_repeated=0, shuffle=False, class_sep=1.5, random_state=0)


@pytest.mark.parametrize("selector", [gwo.gwo_select, gwo.gwo_chaotic_v_select])
@pytest.mark.parametrize("k", [None, 10])
def test_interface(data, selector, k):
    X, y = data
    idx = selector(X, y, k, **QUICK)
    assert isinstance(idx, np.ndarray) and np.issubdtype(idx.dtype, np.integer)
    assert len(idx) == len(np.unique(idx)) > 0
    assert idx.min() >= 0 and idx.max() < X.shape[1]
    if k is not None:
        assert len(idx) == k
    assert gwo.last_history() is not None and len(gwo.last_history()) == QUICK["n_iter"]


def test_fixed_k_pads_when_gwo_selects_too_few(data):
    X, y = data
    model = gwo.BinaryGWO(**QUICK, seed=1).fit(X, y)
    n_free = len(model.selected_idx_)
    padded = gwo.fixed_k(model, n_free + 5)
    assert len(padded) == len(set(padded)) == n_free + 5
    assert set(model.selected_idx_) <= set(padded)


def test_history_is_monotone(data):
    X, y = data
    model = gwo.BinaryGWO(**QUICK).fit(X, y)
    h = np.array(model.history_)
    assert np.all(np.diff(h) <= 1e-12)
    assert model.best_fitness_ == pytest.approx(h[-1])


@pytest.mark.parametrize("selector", [gwo.gwo_select, gwo.gwo_chaotic_v_select])
def test_reproducible(data, selector):
    X, y = data
    a = selector(X, y, None, cache=False, seed=7, **QUICK)
    hist_a = gwo.last_history()
    b = selector(X, y, None, cache=False, seed=7, **QUICK)
    assert np.array_equal(a, b)
    assert hist_a == gwo.last_history()


def test_quality_sanity(data):
    X, y = data
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    def held_out(cols):
        return KNeighborsClassifier(3).fit(Xtr[:, cols], ytr).score(Xte[:, cols], yte)

    # Averaged over seeds: a single stochastic run is too noisy to judge on.
    k, seeds = 10, range(5)
    panels = [gwo.gwo_select(Xtr, ytr, k, seed=s, **QUICK) for s in seeds]
    found = np.mean([len(INFORMATIVE & set(p.tolist())) for p in panels])
    assert found >= 3

    # Control: a random 50% mask over the same prefilter, with the same F-score top-k step.
    rng = np.random.default_rng(0)
    F = gwo.anova_scores(Xtr, ytr)
    top = np.argsort(-F, kind="stable")[: QUICK["prefilter"]]
    control = []
    for _ in range(200):
        m = top[rng.random(len(top)) < 0.5]
        control.append(len(INFORMATIVE & set(m[np.argsort(-F[m])][:k].tolist())))
    assert found > np.mean(control)

    random_acc = np.mean([held_out(rng.choice(X.shape[1], k, replace=False)) for _ in range(50)])
    assert np.mean([held_out(p) for p in panels]) > random_acc


def test_no_leakage_on_random_labels(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 500))
    y = rng.permutation(np.repeat([0, 1], 50))
    selectors = {"GWO": partial(gwo.gwo_select, **QUICK),
                 "GWO-chaotic-V": partial(gwo.gwo_chaotic_v_select, **QUICK)}
    df = run_cv(X, y, selectors, {"kNN-3": KNeighborsClassifier(3)}, k=10, n_splits=5,
                n_repeats=1, dataset_name="noise", out_dir=tmp_path, progress=False)
    assert abs(df["accuracy"].mean() - 0.5) < 0.15


def test_leakage_check_detects_a_leaky_pipeline():
    """Positive control: selecting on the full dataset inflates accuracy beyond the threshold above."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 500))
    y = rng.permutation(np.repeat([0, 1], 50))
    leaky_idx = gwo.gwo_select(StandardScaler().fit_transform(X), y, 10, **QUICK)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    leaky_acc = cross_val_score(KNeighborsClassifier(3), X[:, leaky_idx], y, cv=cv).mean()
    assert leaky_acc - 0.5 > 0.15


def test_frozen_splits(tmp_path, data, monkeypatch):
    X, y = data
    sel = {"GWO": partial(gwo.gwo_select, **QUICK)}
    clf = {"kNN-3": KNeighborsClassifier(3)}
    kw = dict(k=5, n_splits=3, n_repeats=2, dataset_name="toy", out_dir=tmp_path, progress=False)
    first = run_cv(X, y, sel, clf, **kw)
    path = tmp_path / "splits_toy.json"
    saved = path.read_text()

    def must_not_regenerate(*a, **k):
        raise AssertionError("splits were regenerated instead of loaded")

    monkeypatch.setattr(cv_runner, "RepeatedStratifiedKFold", must_not_regenerate)
    second = run_cv(X, y, sel, clf, **kw)
    assert path.read_text() == saved
    folds = json.loads(saved)["folds"]
    assert len(folds) == 6
    for f in folds:
        assert not set(f["train"]) & set(f["test"])
    assert first["selected_genes"].tolist() == second["selected_genes"].tolist()
    assert first["accuracy"].tolist() == second["accuracy"].tolist()


def test_frozen_splits_reject_mismatched_settings(tmp_path, data):
    X, y = data
    kw = dict(k=5, dataset_name="toy", out_dir=tmp_path, progress=False)
    sel = {"GWO": partial(gwo.gwo_select, **QUICK)}
    clf = {"kNN-3": KNeighborsClassifier(3)}
    run_cv(X, y, sel, clf, n_splits=3, n_repeats=1, **kw)
    with pytest.raises(ValueError, match="different settings"):
        run_cv(X, y, sel, clf, n_splits=3, n_repeats=2, **kw)


def test_gwo_only_sees_training_rows(tmp_path, data, monkeypatch):
    X, y = data
    seen = []
    original_fit = gwo.BinaryGWO.fit

    def spy_fit(self, Xf, yf):
        seen.append((np.array(Xf, copy=True), np.array(yf, copy=True)))
        return original_fit(self, Xf, yf)

    monkeypatch.setattr(gwo.BinaryGWO, "fit", spy_fit)
    run_cv(X, y, {"GWO": partial(gwo.gwo_select, **QUICK)}, {"kNN-3": KNeighborsClassifier(3)},
           k=5, n_splits=3, n_repeats=1, dataset_name="toy", out_dir=tmp_path, progress=False)

    folds = json.loads((tmp_path / "splits_toy.json").read_text())["folds"]
    assert len(seen) == len(folds)
    for (Xf, yf), fold in zip(seen, folds):
        train = np.array(fold["train"])
        assert Xf.shape == (len(train), X.shape[1])
        expected = StandardScaler().fit(X[train]).transform(X[train])
        np.testing.assert_allclose(Xf, expected)
        np.testing.assert_array_equal(yf, y[train])


def test_inner_prefilter_removes_inner_selection_bias():
    """On pure noise, the prefilter-biased inner CV looks near-perfect; the honest one does not."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2000))
    y = rng.permutation(np.repeat([0, 1], 30))
    params = dict(n_wolves=10, n_iter=15, prefilter=200)
    honest = gwo.BinaryGWO(**params, inner_prefilter=True).fit(X, y)
    biased = gwo.BinaryGWO(**params, inner_prefilter=False).fit(X, y)
    assert honest.history_[0] > 0.35
    assert biased.best_fitness_ < 0.2


def test_inner_cv_handles_tiny_classes():
    y = np.array([0] * 20 + [1] * 2)
    splits = gwo.inner_splits(y, 3, seed=0)
    assert len(splits) == 2
    y1 = np.array([0] * 20 + [1])
    assert len(gwo.inner_splits(y1, 3, seed=0)) == 3
    rng = np.random.default_rng(0)
    X = rng.normal(size=(21, 50))
    assert len(gwo.gwo_select(X, y1, 5, n_wolves=4, n_iter=3, prefilter=50)) == 5
