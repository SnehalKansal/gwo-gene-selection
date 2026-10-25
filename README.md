# Grey Wolf Optimization for gene selection on microarray cancer data

Does Binary Grey Wolf Optimization (GWO) find better or more stable small gene panels than
standard selectors (ANOVA, LASSO, mRMR, RF importance) on CuMiDa microarray cancer datasets?

Data: CuMiDa curated microarray CSVs (https://sbcb.inf.ufrgs.br/cumida), placed in `data/`
(gitignored). Each CSV has columns `samples`, `type`, then one column per gene.

```bash
pip install -r requirements.txt
```

> `src/data.py` and `src/baselines.py` are Mitali's full implementations (they replaced the
> early placeholders); see "Mitali's part" below.

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

## Mitali's part: data, baselines, stability and enrichment

### Files

| File | Purpose |
|---|---|
| `src/data.py` | `load_cumida(path) -> (X, y, genes)` and `load_cumida_with_meta(path)` (also class names, sample ids, cleaning counts). These validate required columns, duplicate sample ids and ≥2 classes, impute NaNs with column medians, and drop constant columns (both counted). Also provides `class_counts` and `log_transform_if_needed` (log2 only if max > 100). No scaling. |
| `src/validate_data.py` | Week-1 report: `python -m src.validate_data --data ... --name ...` updates `results/dataset_summary.csv` and `.md` (class counts, imbalance, cleaning). Exits 1 with a warning if a class has < 5 samples. |
| `src/baselines.py` | `anova_select`, `lasso_select`, `rf_select`, `mrmr_select` (from scratch, MIQ/MID). All except ANOVA start from the shared `_prefilter` (top 2000 by F-score). |
| `src/classifiers.py` | `get_classifiers(seed)`: the linear SVM, 3-NN and RF-200 used in the results files. |
| `src/stability.py` | Kuncheva, Jaccard and Nogueira (with 95% CI) stability across outer folds, selection frequencies, between-selector overlap, and a stability-vs-k plot. |
| `src/enrichment.py` | Offline GO over-representation analysis: GMT and GEO/GPL annotation loaders, probe-to-symbol mapping, hypergeometric test with BH FDR against the array background, consensus panels from fold frequencies, and an optional Enrichr call via gseapy. |
| `src/tables.py` | Performance tables (mean and 95% t-CI), Wilcoxon paired tests vs a reference selector (Holm-corrected), Friedman mean ranks, and dependency-free markdown export. CLI: `python -m src.tables --name ... --k 30`. |
| `notebooks/stability_analysis.ipynb` | Stability table and plot, selector-overlap heatmap, and most frequently selected genes. |
| `notebooks/go_enrichment.ipynb` | Consensus panels, then probe-to-symbol mapping and GO BP enrichment for each selector, with caveats. |
| `tests/test_mitali.py` | Interface, determinism, statelessness, no-input-modification, train-rows-only and quality tests for every selector. Also known-value checks for stability, ORA and BH, CLI tests, and an integration test through `run_cv`. |
| `docs/report_mitali_draft.md` | Draft of the report's data, baseline, stability and enrichment sections. |

### Selector contract (as implemented)

Every selector returns exactly `k` unique, valid `int64` column indices ordered by importance.
It raises `ValueError` if `k` is not in `[1, p]`, uses only the arrays passed in (no globals, no
caches), never modifies its inputs, and is deterministic given `seed` (default 0).

| Selector | Method | Notes |
|---|---|---|
| `anova_select` | top-k ANOVA F | ranks all genes; NaN-safe |
| `lasso_select(C=None, seed=0, multiclass="ovr")` | L1 logistic on the prefiltered genes, ranked by max \|coef\| across classes, then F | `C=None` picks C from {0.01, 0.1, 1} by inner 3-fold balanced accuracy on the training data. Binary uses liblinear. Multiclass defaults to one-vs-rest liblinear because saga multinomial does not converge on p >> n data (see docstring); `multiclass="multinomial"` uses saga. |
| `rf_select(n_estimators=300, seed=0)` | RF impurity importance on the prefiltered genes | ties by F |
| `mrmr_select(n_candidates=500, scheme="miq")` | greedy mRMR: relevance = F, redundancy = mean \|Pearson r\| with the selected genes | `"miq"`: relevance / redundancy; `"mid"`: relevance/max − redundancy |

### Running

```bash
pytest -q                                          # all tests (Snehal's + Mitali's), ~15 s
python -m src.validate_data --data data/Leukemia_GSE9476.csv --name Leukemia_GSE9476   # week-1 class counts
python run_experiment.py --data data/Leukemia_GSE9476.csv --name Leukemia_GSE9476 --repeats 3 --ks 10 20 30 50
python -m src.tables --name Leukemia_GSE9476 --k 30   # performance / paired tests / ranks -> results/tables/
jupyter notebook notebooks/stability_analysis.ipynb   # set DATASET in the config cell
jupyter notebook notebooks/go_enrichment.ipynb
```

`run_experiment.py` adds mRMR automatically now that `mrmr_select` exists. The enrichment
notebook needs two reference files in `data/ref/` (gitignored):

```bash
mkdir -p data/ref
curl -L -o data/ref/GPL96.annot.gz https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz
curl -L -o data/ref/GO_Biological_Process_2023.gmt "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Biological_Process_2023"
```

GPL96 (HG-U133A) is the platform of GSE9476. Other datasets need their own GPL annotation
(see the GEO series page). The notebooks also read the environment variables `GWO_DATASET`,
`GWO_RESULTS_DIR`, `GWO_K`, `GWO_ANNOTATION` and `GWO_GMT`.

### Rigor notes

- Stability and enrichment only use `selected_genes` from the results file. Each list was chosen
  on an outer training fold, and enrichment panels are genes selected in at least 60% of folds.
- The enrichment background is the set of genes measured on the array, not the genome.
- NaN imputation and constant-column removal in `data.py` use no labels. Median imputation does
  use all samples (CuMiDa files normally contain no NaNs; the count is reported).
