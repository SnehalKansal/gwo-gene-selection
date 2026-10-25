"""Binary Grey Wolf Optimization (GWO) for wrapper-based gene selection.

All fitting happens on whatever data is passed in. When called from ``cv_runner`` that
data is the (already scaled) outer training fold only; the prefilter, the inner CV used
by the fitness function, and the final ranking never see test-fold rows.
"""
from __future__ import annotations

import hashlib
import time
import warnings
from collections import OrderedDict
from typing import Any

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC

POSITION_BOUND = 6.0
INIT_RANGE = (-4.0, 4.0)
IMBALANCE_RATIO = 1.5
CACHE_SIZE = 64

_LAST_INFO: dict[str, Any] | None = None
_FIT_CACHE: OrderedDict[str, "BinaryGWO"] = OrderedDict()


def anova_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ANOVA F-score per column; constant columns get 0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F, _ = f_classif(X, y)
    return np.nan_to_num(F, nan=0.0)


def inner_splits(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified inner splits with ``n_splits`` capped by the smallest class size.

    Falls back to plain shuffled KFold when the smallest class has a single sample.
    """
    _, counts = np.unique(y, return_counts=True)
    n = int(min(n_splits, counts.min()))
    dummy = np.zeros(len(y))
    if n >= 2:
        return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(dummy, y))
    n = int(min(max(n_splits, 2), len(y)))
    return list(KFold(n, shuffle=True, random_state=seed).split(dummy))


class BinaryGWO:
    """Binary Grey Wolf Optimizer over an ANOVA-prefiltered gene set.

    Fitness (minimised) is ``(1 - mean inner-CV score) + alpha_penalty * n_selected / n_candidates``,
    where the score is balanced accuracy if the largest/smallest class ratio exceeds
    ``IMBALANCE_RATIO`` and plain accuracy otherwise. An empty mask has fitness 1.0.

    ``inner_prefilter=True`` (default) keeps the inner CV honest: each inner fold recomputes
    the ANOVA top-``prefilter`` list on its own inner-training rows and evaluates only
    ``mask & that list``, so inner-validation rows never influence which genes they are
    scored on. The mask is defined over the union of the global top-``prefilter`` genes and
    all inner-fold lists (``candidate_idx_``). With ``inner_prefilter=False`` the mask covers
    only the global list and every inner fold uses it directly; the inner score is then
    optimistically biased (the prefilter has already seen the inner-validation rows).
    """

    def __init__(
        self,
        n_wolves: int = 20,
        n_iter: int = 40,
        alpha_penalty: float = 0.01,
        prefilter: int = 2000,
        transfer: str = "s",
        init: str = "random",
        classifier: str = "knn",
        inner_cv: int = 3,
        seed: int = 0,
        inner_prefilter: bool = True,
    ) -> None:
        if transfer not in ("s", "v"):
            raise ValueError("transfer must be 's' or 'v'")
        if init not in ("random", "chaotic"):
            raise ValueError("init must be 'random' or 'chaotic'")
        if classifier not in ("knn", "svm"):
            raise ValueError("classifier must be 'knn' or 'svm'")
        self.n_wolves = n_wolves
        self.n_iter = n_iter
        self.alpha_penalty = alpha_penalty
        self.prefilter = prefilter
        self.transfer = transfer
        self.init = init
        self.classifier = classifier
        self.inner_cv = inner_cv
        self.seed = seed
        self.inner_prefilter = inner_prefilter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinaryGWO":
        """Run the optimisation on (X, y), which must be training data only."""
        start = time.perf_counter()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)

        self.f_scores_ = anova_scores(X, y)
        n_keep = int(min(self.prefilter, X.shape[1]))
        self.prefilter_idx_ = np.argsort(-self.f_scores_, kind="stable")[:n_keep]
        splits = inner_splits(y, self.inner_cv, self.seed)
        fold_tops = [np.argsort(-anova_scores(X[tr], y[tr]), kind="stable")[:n_keep]
                     for tr, _ in splits] if self.inner_prefilter else []
        extra = np.setdiff1d(np.concatenate([self.prefilter_idx_, *fold_tops]), self.prefilter_idx_)
        extra = extra[np.argsort(-self.f_scores_[extra], kind="stable")]
        self.candidate_idx_ = np.concatenate([self.prefilter_idx_, extra]).astype(int)
        n_cand = len(self.candidate_idx_)
        Xp = X[:, self.candidate_idx_]

        _, counts = np.unique(y, return_counts=True)
        self.use_balanced_ = bool(counts.max() / counts.min() > IMBALANCE_RATIO)
        self._folds = []
        for i, (tr, va) in enumerate(splits):
            mu, sd = Xp[tr].mean(axis=0), Xp[tr].std(axis=0)
            sd[sd == 0] = 1.0
            allowed = (np.isin(self.candidate_idx_, fold_tops[i]) if self.inner_prefilter
                       else np.ones(n_cand, dtype=bool))
            self._folds.append(((Xp[tr] - mu) / sd, y[tr], (Xp[va] - mu) / sd, y[va], allowed))
        self._cache: dict[bytes, float] = {}

        shape = (self.n_wolves, n_cand)
        pos = self._init_positions(rng, shape)
        masks = self._sigmoid_masks(pos, rng)
        lead_pos = np.zeros((3, n_cand))
        lead_mask = np.zeros((3, n_cand), dtype=bool)
        lead_fit = np.full(3, np.inf)
        self.history_: list[float] = []

        for t in range(self.n_iter):
            if t > 0:
                masks = self._binarize(pos, masks, rng)
            fits = self._population_fitness(masks)
            lead_pos, lead_mask, lead_fit = self._update_leaders(
                lead_pos, lead_mask, lead_fit, pos, masks, fits
            )
            self.history_.append(float(lead_fit[0]))
            a = 2.0 * (1.0 - t / max(self.n_iter - 1, 1))
            r1, r2 = rng.random((3, *shape)), rng.random((3, *shape))
            A, C = 2.0 * a * r1 - a, 2.0 * r2
            D = np.abs(C * lead_pos[:, None, :] - pos[None])
            pos = np.clip((lead_pos[:, None, :] - A * D).mean(axis=0), -POSITION_BOUND, POSITION_BOUND)

        self.best_mask_ = lead_mask[0].copy()
        self.best_fitness_ = float(lead_fit[0])
        chosen = self.candidate_idx_[self.best_mask_]
        self.selected_idx_ = chosen[np.argsort(-self.f_scores_[chosen], kind="stable")].astype(int)
        self.n_evaluations_ = len(self._cache)
        self.fit_time_ = time.perf_counter() - start
        del self._folds, self._cache
        return self

    def _init_positions(self, rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
        lo, hi = INIT_RANGE
        if self.init == "random":
            u = rng.random(shape)
        else:
            u = np.empty(shape)
            x = rng.uniform(0.01, 0.99, size=shape[1])
            for _ in range(10):
                x = 4.0 * x * (1.0 - x)
            for w in range(shape[0]):
                x = 4.0 * x * (1.0 - x)
                u[w] = x
        return lo + (hi - lo) * u

    @staticmethod
    def _sigmoid_masks(pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.random(pos.shape) < 1.0 / (1.0 + np.exp(-pos))

    def _binarize(self, pos: np.ndarray, prev: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.transfer == "s":
            return self._sigmoid_masks(pos, rng)
        flip = rng.random(pos.shape) < np.abs(np.tanh(pos))
        return np.where(flip, ~prev, prev)

    @staticmethod
    def _update_leaders(lead_pos, lead_mask, lead_fit, pos, masks, fits):
        all_fit = np.concatenate([lead_fit, fits])
        all_pos = np.vstack([lead_pos, pos])
        all_mask = np.vstack([lead_mask, masks])
        picked, seen = [], set()
        for i in np.argsort(all_fit, kind="stable"):
            key = np.packbits(all_mask[i]).tobytes()
            if np.isfinite(all_fit[i]) and key not in seen:
                seen.add(key)
                picked.append(i)
            if len(picked) == 3:
                break
        while len(picked) < 3:
            picked.append(picked[-1])
        return all_pos[picked].copy(), all_mask[picked].copy(), all_fit[picked].copy()

    def _population_fitness(self, masks: np.ndarray) -> np.ndarray:
        """Fitness of every wolf, evaluating only masks not already in the run cache."""
        keys = [np.packbits(m).tobytes() for m in masks]
        todo = [i for i, key in enumerate(keys) if key not in self._cache]
        todo = list({keys[i]: i for i in todo}.values())
        if todo:
            for i, fit in zip(todo, self._evaluate(masks[todo])):
                self._cache[keys[i]] = fit
        return np.array([self._cache[key] for key in keys])

    def _evaluate(self, masks: np.ndarray) -> np.ndarray:
        n_sel = masks.sum(axis=1)
        acc = np.zeros(len(masks))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for Xtr, ytr, Xva, yva, allowed in self._folds:
                eff = masks & allowed
                classes, counts = np.unique(ytr, return_counts=True)
                majority = classes[counts.argmax()]
                if len(classes) < 2:
                    preds = np.full((len(masks), len(yva)), majority)
                elif self.classifier == "knn":
                    preds = _knn_predict_batch(Xtr, ytr, Xva, eff)
                else:
                    preds = np.array([
                        SVC(kernel="linear", C=1.0).fit(Xtr[:, m], ytr).predict(Xva[:, m])
                        if m.any() else np.full(len(yva), majority)
                        for m in eff
                    ])
                preds[~eff.any(axis=1)] = majority
                acc += _scores(yva, preds, self.use_balanced_)
        acc /= len(self._folds)
        fit = (1.0 - acc) + self.alpha_penalty * n_sel / masks.shape[1]
        fit[n_sel == 0] = 1.0
        return fit

    def get_params(self) -> dict[str, Any]:
        """Constructor parameters as a dict."""
        keys = ("n_wolves", "n_iter", "alpha_penalty", "prefilter", "transfer", "init",
                "classifier", "inner_cv", "seed", "inner_prefilter")
        return {k: getattr(self, k) for k in keys}


def _scores(y_true: np.ndarray, preds: np.ndarray, balanced: bool) -> np.ndarray:
    """Accuracy (or balanced accuracy: mean recall over classes present in y_true) per row of preds."""
    hit = preds == y_true[None, :]
    if not balanced:
        return hit.mean(axis=1)
    classes = np.unique(y_true)
    return np.mean([hit[:, y_true == c].mean(axis=1) for c in classes], axis=0)


def _knn_predict_batch(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, masks: np.ndarray,
                       n_neighbors: int = 3) -> np.ndarray:
    """3-NN (Euclidean, majority vote, ties to the nearest neighbour) for many gene masks at once.

    Returns predictions of shape (n_masks, n_val). Squared distances restricted to a mask
    are ``|a|^2_m + |b|^2_m - 2 a diag(m) b^T``, so every mask is a weighted matmul.
    """
    W = masks.astype(float)
    sq_tr = (Xtr ** 2) @ W.T
    sq_va = (Xva ** 2) @ W.T
    cross = (Xva[None, :, :] * W[:, None, :]) @ Xtr.T
    dist = sq_va.T[:, :, None] + sq_tr.T[:, None, :] - 2.0 * cross
    k = min(n_neighbors, len(ytr))
    nn = np.argsort(dist, axis=2, kind="stable")[:, :, :k]
    classes, codes = np.unique(ytr, return_inverse=True)
    nn_codes = codes[nn]
    votes = (nn_codes[..., None] == np.arange(len(classes))).sum(axis=2).astype(float)
    return classes[votes.argmax(axis=2)]


def _fit_key(X: np.ndarray, y: np.ndarray, params: dict[str, Any]) -> str:
    classes, codes = np.unique(y, return_inverse=True)
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(X, dtype=float).tobytes())
    h.update(str(X.shape).encode())
    h.update(codes.astype(np.int64).tobytes())
    h.update(repr(classes.tolist()).encode())
    h.update(repr(sorted(params.items())).encode())
    return h.hexdigest()


def _fit(X: np.ndarray, y: np.ndarray, params: dict[str, Any], cache: bool) -> tuple[BinaryGWO, bool]:
    model = BinaryGWO(**params)
    key = _fit_key(np.asarray(X), np.asarray(y), model.get_params())
    if cache and key in _FIT_CACHE:
        _FIT_CACHE.move_to_end(key)
        return _FIT_CACHE[key], True
    model.fit(X, y)
    if cache:
        _FIT_CACHE[key] = model
        while len(_FIT_CACHE) > CACHE_SIZE:
            _FIT_CACHE.popitem(last=False)
    return model, False


def fixed_k(model: BinaryGWO, k: int) -> np.ndarray:
    """Reduce or pad a fitted model's selection to exactly ``k`` genes (see ``gwo_select``)."""
    chosen = model.selected_idx_[:k]
    if len(chosen) < k:
        pool = model.prefilter_idx_[~np.isin(model.prefilter_idx_, chosen)]
        chosen = np.concatenate([chosen, pool[: k - len(chosen)]])
    if len(chosen) < k:
        rest = np.argsort(-model.f_scores_, kind="stable")
        rest = rest[~np.isin(rest, chosen)]
        chosen = np.concatenate([chosen, rest[: k - len(chosen)]])
    return chosen.astype(int)


def gwo_select(X: np.ndarray, y: np.ndarray, k: int | None = None, cache: bool = True,
               **kwargs: Any) -> np.ndarray:
    """Select genes with Binary GWO; follows the shared ``select_genes(X, y, k)`` interface.

    Free mode (``k=None``): returns every gene in GWO's best mask, so the panel size is
    whatever the optimiser found.

    Fixed-k mode (``k`` given, for equal-size comparison with baselines): the genes GWO
    selected are ranked by ANOVA F-score (computed on the same training data) and the top
    ``k`` kept. If GWO selected fewer than ``k`` genes, the panel is padded with the
    next-best genes by F-score from the prefiltered set (then from all genes if needed).

    Extra keyword arguments are passed to ``BinaryGWO``. Fits are memoised on
    (data, labels, parameters) so a k-sweep over the same fold runs GWO only once; pass
    ``cache=False`` to force a fresh run. Returned indices are column indices of ``X``,
    ordered by decreasing F-score.
    """
    global _LAST_INFO
    model, was_cached = _fit(X, y, kwargs, cache)
    selected = model.selected_idx_.copy() if k is None else fixed_k(model, int(k))
    _LAST_INFO = {
        "history": list(model.history_),
        "fit_time_sec": model.fit_time_,
        "cached": was_cached,
        "best_fitness": model.best_fitness_,
        "n_selected_free": int(model.best_mask_.sum()),
    }
    return selected


def gwo_chaotic_v_select(X: np.ndarray, y: np.ndarray, k: int | None = None,
                         **kwargs: Any) -> np.ndarray:
    """``gwo_select`` with logistic-map initialisation and the V-shaped transfer function."""
    kwargs.update(init="chaotic", transfer="v")
    return gwo_select(X, y, k, **kwargs)


def last_history() -> list[float] | None:
    """Best-fitness-per-iteration curve of the most recent GWO selector call (None if reset)."""
    return None if _LAST_INFO is None else list(_LAST_INFO["history"])


def last_info() -> dict[str, Any] | None:
    """History, real optimisation time, cache flag and free-mode size of the last GWO call."""
    return None if _LAST_INFO is None else dict(_LAST_INFO)


def reset_last() -> None:
    """Forget the last call's info (cv_runner calls this before every selector)."""
    global _LAST_INFO
    _LAST_INFO = None


def clear_cache() -> None:
    """Drop all memoised GWO fits."""
    _FIT_CACHE.clear()
