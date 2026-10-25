"""Baseline gene selectors. Every selector follows ``select_genes(X_train, y_train, k)``."""
from __future__ import annotations

import warnings

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def _anova_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F, _ = f_classif(X, y)
    return np.nan_to_num(F, nan=0.0)


def _top_k(scores: np.ndarray, k: int, tiebreak: np.ndarray) -> np.ndarray:
    order = np.lexsort((-tiebreak, -scores))
    return order[: min(k, len(order))].astype(int)


def anova_select(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Top-k genes by ANOVA F-score."""
    F = _anova_scores(X, y)
    return _top_k(F, k, F)


def lasso_select(X: np.ndarray, y: np.ndarray, k: int, C: float = 0.5, seed: int = 0) -> np.ndarray:
    """Top-k genes by one-vs-rest L1-logistic coefficient magnitude (max over classes), ties by F-score."""
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    l1 = {"l1_ratio": 1.0} if (major, minor) >= (1, 8) else {"penalty": "l1"}
    base = LogisticRegression(solver="liblinear", C=C, max_iter=2000, random_state=seed, **l1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = OneVsRestClassifier(base).fit(X, y)
    coef = np.abs(np.vstack([est.coef_ for est in model.estimators_])).max(axis=0)
    return _top_k(coef, k, _anova_scores(X, y))


def rf_select(X: np.ndarray, y: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Top-k genes by random-forest impurity importance."""
    rf = RandomForestClassifier(n_estimators=200, random_state=seed).fit(X, y)
    return _top_k(rf.feature_importances_, k, _anova_scores(X, y))
