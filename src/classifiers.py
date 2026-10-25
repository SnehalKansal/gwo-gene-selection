"""Downstream classifiers, with the same names and settings as ``run_experiment.py``."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def get_classifiers(seed: int = 0) -> dict[str, object]:
    """Unfitted classifiers keyed by the names used in the results files."""
    return {
        "SVM-linear": SVC(kernel="linear", C=1.0),
        "kNN-3": KNeighborsClassifier(n_neighbors=3),
        "RF-200": RandomForestClassifier(n_estimators=200, random_state=seed),
    }
