# Draft: Data, Baselines, Stability and Enrichment (Mitali)

*Placeholders in `[[double brackets]]` are filled from `results/` after the full runs.
Pilot figures quoted below come from a quick run (5 folds × 1 repeat, reduced GWO settings) and
will be replaced.*

## Data

We used curated microarray datasets from CuMiDa (Feltes et al., 2019), a benchmark collection
of cancer gene-expression series from GEO, re-processed and quality-filtered for machine
learning. Each dataset is a samples × probes matrix with one class label per sample. Columns
are Affymetrix probe set identifiers (e.g. `1007_s_at`), not gene symbols. The datasets were
[[list: Leukemia_GSE9476, …]].

**Loading and validation** (`src/data.py`). For each file we check that the `samples` and
`type` columns exist, that sample identifiers are unique, and that there are at least two
classes. Non-numeric values are treated as missing and imputed with the column median, and
columns that are constant across all samples are dropped. Both counts are reported.
Neither step uses class labels. Expression values are kept as provided: CuMiDa data are
already on a log2 scale, so the raw-intensity check (log2-transform only if the maximum
exceeds 100) did not trigger for any dataset. No scaling is done at load time; standardisation
is fitted inside each outer training fold (Snehal's Methods).

**Class composition** was tabulated for every dataset before any modelling
(`python -m src.validate_data`; Table [[D1]]).

**Table [[D1]].** Dataset summary.

| Dataset | Samples | Probes | Classes | Class counts | Max/min ratio | Dropped constant | NaNs imputed |
|---|---|---|---|---|---|---|---|
| Leukemia_GSE9476 | 64 | 22,283 | 5 | AML 26, Bone marrow 10, Bone marrow CD34+ 8, Peripheral blood 10, PBSC CD34+ 10 | 3.25 | 0 | 0 |
| [[…]] | | | | | | | |

Leukemia_GSE9476 has five classes. AML samples are contrasted with four normal haematopoietic
compartments, and the smallest class has 8 samples. This limits outer CV to at most 8
folds; the main protocol uses 5. With 5 outer folds there are about 6 samples of the smallest
class per training fold, which is enough for GWO's 3-fold inner CV, though only just.
[[Note any dataset with a class < 5 samples; `validate_data` flags these.]]

## Baseline selectors

All selectors share one interface, `select_genes(X_train, y_train, k)`. Each receives only the
standardised outer training fold and returns exactly $k$ unique column indices, ordered by
importance. None keeps state between calls or modifies its inputs, and all are deterministic
given a seed (default 0). Automated tests check each of these properties, including that
changing rows outside the training fold cannot change a selection. Except for ANOVA itself,
every selector first reduces the $p \approx 22{,}000$ probes to the 2,000 with the highest ANOVA
F-statistic on the training fold.

- **ANOVA F-test.** The $k$ probes with the largest one-way ANOVA F-statistic across classes.
  This is a univariate filter that ignores redundancy.
- **LASSO.** L1-penalised logistic regression on the prefiltered probes. Probes are ranked by
  their largest absolute coefficient across classes, with F-score breaking ties, so the panel
  is filled by F-score when fewer than $k$ coefficients are non-zero. The inverse regularisation
  strength $C \in \{0.01, 0.1, 1\}$ is chosen by stratified 3-fold inner CV (balanced accuracy)
  on the training fold. Binary problems use liblinear. For multiclass problems we fit one
  one-vs-rest L1 model per class with liblinear, rather than a multinomial model with saga.
  In pilot runs on Leukemia_GSE9476, saga stopped far from the L1 optimum at any practical
  tolerance: 493 non-zero coefficients at tol $10^{-3}$ versus 58 at $10^{-5}$, and the top-30
  probes changed by more than half. One-vs-rest liblinear agreed with the near-converged
  multinomial solution on 22 of the top 30 probes, in 0.03 s.
- **Random-forest importance.** Mean decrease in impurity from a 300-tree random forest on
  the prefiltered probes (seeded), with ties broken by F-score.
- **mRMR** (Peng et al., 2005), implemented from scratch. Relevance is the ANOVA F-statistic,
  and redundancy is a candidate's mean absolute Pearson correlation with the probes already
  chosen. Starting from the most relevant probe, the method greedily adds the candidate that
  maximises relevance / (redundancy + ε) (the MIQ scheme, our default). The MID scheme
  (relevance rescaled to [0, 1] minus redundancy) is also available. The search covers the 500
  most relevant prefiltered probes. Because redundancy is a mean, a highly relevant probe can
  re-enter once the panel is large even if it is correlated with one selected probe. Our tests
  confirm that mRMR never takes a near-duplicate as its second pick, while ANOVA takes all
  duplicates.

No selector hyperparameter was tuned on outer-fold performance. Prefilter size, C grid, forest
size and mRMR candidate pool were fixed in advance and are identical across datasets.

## Stability analysis

A panel that changes completely when a few samples change is hard to interpret biologically,
so we measure how consistent each selector is across the 15 outer training folds
(`src/stability.py`). The analysis uses only the gene lists the selectors produced inside the
training folds; no test-fold information enters. Three indices are reported:

- **Kuncheva index** (Kuncheva, 2007). The mean over fold pairs of
  $(|A \cap B| - k^2/p)/(k - k^2/p)$, which corrects the overlap for chance; it requires
  equal panel sizes.
- **Jaccard similarity.** The mean over fold pairs of $|A \cap B|/|A \cup B|$. It is intuitive
  but not chance-corrected.
- **Nogueira stability** (Nogueira et al., 2018). $\hat\Phi = 1 - \overline{s_f^2} / [\bar k/p\,(1 - \bar k/p)]$,
  computed from per-probe selection frequencies, with an asymptotic 95% confidence interval.
  It handles variable panel sizes (needed for free-mode GWO) and equals the Kuncheva index
  when sizes are equal, which our tests verify. $\hat\Phi = 1$ means identical panels;
  $\hat\Phi \approx 0$ means agreement no better than random panels of the same size.

We also report the mean per-fold Jaccard overlap between pairs of selectors, and the probes
selected most often by each method.

## Exploratory GO enrichment

To get a rough view of the biology each method picks up, we ran an over-representation
analysis (`src/enrichment.py`) on *consensus panels*: probes selected in at least 60% of the
outer training folds at $k = 30$. Probes were mapped to HGNC symbols with the GEO platform
annotation (GPL96 for GSE9476; multi-gene probes were split on `///`, and unmapped probes were
dropped and counted). Each GO Biological Process set (GO_Biological_Process_2023; [[n]] sets
with 5–500 genes after restriction) was tested with a one-sided hypergeometric test. The
background is all genes measured on the array (13,908 symbols for GPL96), not the whole genome,
which avoids the inflated significance a genome background produces. P-values were adjusted with
Benjamini–Hochberg; we call q < 0.05 enriched. This analysis is exploratory and
hypothesis-generating; it does not validate the panels.

## Results skeleton

### Baselines vs GWO

See Snehal's Table [[1]] for accuracy, macro-F1 and minimum recall. Table [[B2]]
(`python -m src.tables`) gives Friedman mean ranks across folds and classifiers, and Wilcoxon
signed-rank tests of GWO against each baseline (Holm-corrected within classifier). Folds are not
independent, so these p-values are descriptive. [[In the pilot run (k = 30, 5 folds) the mean
ranks were mRMR 2.9, GWO 3.3, LASSO 3.3, RF 3.5, ANOVA 3.8, GWO-chaotic-V 4.2, with Friedman
p = 0.08. Replace with the full run.]]

### Stability

**Table [[S1]].** Kuncheva, Jaccard and Nogueira stability (95% CI) per selector at $k = 30$.

| Selector | Kuncheva | Jaccard | Nogueira [95% CI] |
|---|---|---|---|
| ANOVA | [[ ]] | [[ ]] | [[ ]] |
| LASSO | [[ ]] | [[ ]] | [[ ]] |
| RF | [[ ]] | [[ ]] | [[ ]] |
| mRMR | [[ ]] | [[ ]] | [[ ]] |
| GWO | [[ ]] | [[ ]] | [[ ]] |
| GWO-chaotic-V | [[ ]] | [[ ]] | [[ ]] |
| GWO-free | n/a | [[ ]] | [[ ]] |

Figure [[S2]] (`figures/stability_<dataset>.png`) shows stability against $k$, and Figure [[S3]]
(`figures/selector_overlap_<dataset>.png`) shows between-selector overlap. [[Pilot
(Leukemia_GSE9476, k = 30, 5 folds): Nogueira 0.80 for ANOVA, 0.70 for mRMR, 0.53 for LASSO,
0.45 for GWO-chaotic-V, 0.32 for GWO-free, 0.28 for GWO and 0.25 for RF. In that run the
filter methods were clearly more stable than GWO, and GWO and RF panels overlapped little with
each other or with the filters. Confirm with the full run.]] [[Interpret: is GWO's accuracy,
if comparable, bought with less stable panels?]]

### Enrichment

**Table [[E1]].** For each selector: consensus probes, mapped symbols, sets tested, sets with
q < 0.05, and the top term. [[Pilot: 13–26 consensus probes per selector, with no GO BP term
at q < 0.05 for any selector (smallest q 0.13, mRMR). Report the full-run result as it is;
a null result is expected for panels this small.]] [[If terms are enriched, list them and note
which are immune or cell-type terms (see Limitations).]]

## Role and responsibilities (Mitali)

Mitali implemented data loading and validation, the week-1 dataset summary, the four baseline
selectors (ANOVA, LASSO, random-forest importance and a from-scratch mRMR) and the shared
classifier set. She also built the stability analysis (Kuncheva, Jaccard and Nogueira indices
with confidence intervals), the offline GO enrichment pipeline with probe-to-gene mapping, the
results tables and paired tests, and the analysis notebooks and automated tests for these
modules. She wrote the Data, Baselines, Stability and Enrichment sections of this report. The
GWO algorithm, the nested cross-validation runner and the performance figures were done by
Snehal.

## Limitations

- **Exploratory biology.** The enrichment results generate hypotheses and do not validate
  anything. Consensus panels of 10–30 probes give little power, the result depends on the
  60% frequency threshold and on the GO release used, and a gene appearing in a panel does not
  show that it drives the disease.
- **Immune infiltration and cell-type composition.** Several CuMiDa contrasts compare tumour
  with tissue of different cellular makeup (for GSE9476, AML against normal marrow, blood and
  CD34+ fractions). Selected genes, and any immune or haematopoietic GO terms, may therefore
  reflect which cells were sampled rather than tumour-intrinsic programmes.
- **Batch effects.** Samples in a GEO series were often processed in batches (scan date, lab,
  protocol) that can correlate with class. Selectors may pick probes that track batch rather
  than biology. We did not correct for batch, since the batch annotation is usually not
  available in CuMiDa.
- **Probe-level artifacts.** Probe sets are not genes. Cross-hybridising `_x_at`/`_s_at`
  probes, probes that map to several genes (`///`), several probes per gene, and the roughly 5%
  of GPL96 probes with no symbol all blur the gene-level picture. Stability is computed at probe
  level, so two probes for the same gene count as different features.
- **Small n and fold dependence.** With 64 samples, the 15 outer folds share most of their
  training data. Stability across folds therefore overstates stability across independent
  cohorts, and confidence intervals and paired-test p-values are optimistic.
- **Shared prefilter.** All selectors except ANOVA, and GWO itself, search only the top 2,000
  probes by F-score. This favours marginally strong probes, and some of the agreement between
  methods comes from this shared step.
- **Preprocessing on the full dataset.** Median imputation uses all samples. This uses no
  labels, but it is not fitted per fold; it was a no-op for every dataset here (0 NaNs).
