# Grey Wolf Optimization for gene selection on microarray cancer data

Does Binary Grey Wolf Optimization (GWO) find better or more stable small gene panels than
standard selectors (ANOVA, LASSO, mRMR, RF importance) on CuMiDa microarray cancer datasets?

Data: CuMiDa curated microarray CSVs (https://sbcb.inf.ufrgs.br/cumida), placed in `data/`
(gitignored). Each CSV has columns `samples`, `type`, then one column per gene.

```bash
pip install -r requirements.txt
```

> **Placeholder note.** `src/data.py` and `src/baselines.py` are owned by Mitali. The current
> versions are minimal placeholders written so the GWO pipeline runs end to end
> (`load_cumida`, `anova_select`, `lasso_select`, `rf_select`; no `mrmr_select` yet).
> Mitali will replace them; nothing else needs to change as long as the interface below is kept.

## Snehal's part: GWO and CV runner

### Files

| File | Purpose |
|---|---|
| `src/gwo.py` | Binary Grey Wolf Optimizer written from scratch (`BinaryGWO`), plus the selectors `gwo_select` and `gwo_chaotic_v_select` and the `last_history()` helper used to log convergence. |
| `src/cv_runner.py` | `run_cv(...)`: frozen repeated stratified outer CV. Fold-wise scaling and selection, metrics, and saved results, splits and convergence curves. `load_results(...)` reads them back. |
| `src/plots.py` | `plot_convergence`, `plot_accuracy_vs_genes`, `plot_selector_comparison`. Write PNGs (dpi 200) to `<out>/figures/`. |
| `run_experiment.py` | CLI that runs every selector × classifier, writes results and figures, and prints a summary table. |
| `tests/test_snehal.py` | Pytest suite: interface, reproducibility, quality sanity, leakage (plus a positive control), frozen splits, train-only fitting, inner-CV bias, tiny classes. |
| `docs/report_snehal_draft.md` | Draft of the report's GWO Methods and Results sections. |

### Shared interface contract

Every selector must have the signature

```python
select_genes(X_train, y_train, k) -> np.ndarray  # integer column indices into X_train
```

- `X_train` is the outer training fold, already standardised by a scaler fit on that fold.
  Selectors never see test rows.
- Indices must be unique, valid column indices, and exactly `k` of them. The only exception
  is `GWO-free`, which ignores `k` and returns whatever GWO selected.
- Extra parameters are bound with `functools.partial`, e.g.
  `partial(gwo_select, n_wolves=10, n_iter=15)`.
- `src.data.load_cumida(path)` returns `(X, y, gene_names, class_names)`; only `X` and `y`
  are used here.

### Running

```bash
# tests (about 10 s)
pytest tests/test_snehal.py -q

# quick smoke test on one dataset (separate out dir so its 1-repeat splits don't clash)
python run_experiment.py --data data/Leukemia_GSE9476.csv --name Leukemia_GSE9476 --quick --out results_smoke

# quick smoke test without any data (synthetic CuMiDa-shaped data: 72 samples x 5000 genes, 3 classes)
python run_experiment.py --synthetic --quick --ks 10 20 30 50 --out results_smoke

# full experiment: 5-fold x 3 repeats, default GWO settings, k-sweep for the accuracy-vs-genes figure
python run_experiment.py --data data/Leukemia_GSE9476.csv --name Leukemia_GSE9476 --repeats 3 --ks 10 20 30 50
```

Outputs in `--out` (default `results/`):

- `splits_<name>.json`: frozen outer folds. They are reused on later runs, and a run with
  different CV settings is refused rather than silently resplitting.
- `results_<name>.csv`: one row per (repeat, fold, selector, classifier, k), with accuracy,
  macro-F1, balanced accuracy, min recall, per-class recall, n_genes, selected gene indices
  (for the stability analysis), selection time and GWO history. List columns are JSON strings;
  `src.cv_runner.load_results` decodes them.
- `convergence_<name>.json`: GWO best-fitness curve per selector and fold.
- `figures/convergence_<name>.png`, `figures/selector_comparison_<name>.png`,
  `figures/accuracy_vs_genes_<name>.png` (the last one only when several k are run).

Selectors run by `run_experiment.py`: ANOVA, LASSO, RF (plus mRMR once `mrmr_select`
exists), GWO, GWO-chaotic-V (logistic-map init + V-shaped transfer), and GWO-free (GWO's
own panel size). Classifiers: linear SVM, kNN (k=3), random forest (200 trees). `--quick`
sets GWO to 10 wolves, 15 iterations and a 1000-gene prefilter.

### Rigor controls in the code

- The scaler, ANOVA prefilter and all selectors are fit on the outer training fold only
  (`test_gwo_only_sees_training_rows` checks that GWO receives exactly the scaled training rows).
- GWO's fitness uses a stratified 3-fold inner CV on the training fold, capped by the smallest
  class size. Each inner fold re-standardises on its inner-train rows and, by default, redoes
  the ANOVA prefilter on those rows (`inner_prefilter=True`). Without this, the inner score is
  near-perfect even on pure noise, because the prefilter has already seen the inner-validation
  rows.
- Outer CV is `RepeatedStratifiedKFold(random_state=42)` with folds frozen to JSON. All GWO
  runs are seeded (`seed=0` by default).
- On random labels the full pipeline scores at chance (`test_no_leakage_on_random_labels`).
  Selecting on the full dataset scores well above chance (`test_leakage_check_detects_a_leaky_pipeline`).
