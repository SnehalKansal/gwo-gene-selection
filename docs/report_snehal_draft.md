# Draft: GWO Methods and Results (Snehal)

*Placeholders in `[[double brackets]]` are filled from `results/` after the full runs.*

## Methods

### Binary Grey Wolf Optimization

Grey Wolf Optimization (GWO; Mirjalili et al., 2014) is a population-based metaheuristic
modelled on the leadership hierarchy and hunting behaviour of grey wolves. We implemented a
binary variant for gene selection from scratch (`src/gwo.py`). A population of
$N$ wolves is kept, each with a continuous position vector $\mathbf{x}_i \in [-6, 6]^d$
over a candidate gene set of size $d$. At each iteration every position is mapped to a binary
mask $\mathbf{m}_i \in \{0,1\}^d$ (gene included or excluded), and masks are scored by the
fitness function below. The three best solutions found so far form the leaders $\alpha$,
$\beta$ and $\delta$. They are kept across iterations, so the best fitness never increases.
Every wolf then moves towards the leaders:

$$
\mathbf{D}_\ell = |\mathbf{C}_\ell \odot \mathbf{x}_\ell - \mathbf{x}_i|,\quad
\mathbf{X}_\ell = \mathbf{x}_\ell - \mathbf{A}_\ell \odot \mathbf{D}_\ell,\quad
\mathbf{x}_i \leftarrow \tfrac{1}{3}(\mathbf{X}_\alpha + \mathbf{X}_\beta + \mathbf{X}_\delta),
$$

with $\mathbf{A} = 2a\mathbf{r}_1 - a$, $\mathbf{C} = 2\mathbf{r}_2$, and
$\mathbf{r}_1, \mathbf{r}_2 \sim U(0,1)^d$ drawn independently per leader and wolf. The
coefficient $a$ decreases linearly from 2 to 0 over the run, moving the search from
exploration ($|A| > 1$) to exploitation ($|A| < 1$).

**Transfer functions.** Two maps from continuous positions to masks were compared.
The *S-shaped* (sigmoid) rule samples each bit independently:
$m_j = 1$ if $r < 1/(1 + e^{-x_j})$. The *V-shaped* rule uses $T(x) = |\tanh(x)|$ as a
bit-flip probability: bit $j$ of the wolf's previous mask is flipped with probability
$T(x_j)$ and kept otherwise. The first iteration always uses the sigmoid rule, so the
V-shaped variant has an initial mask to flip.

**Initialisation.** Positions are drawn uniformly on $[-4, 4]$ (*random*). The *chaotic*
variant uses the logistic map $z_{n+1} = 4z_n(1 - z_n)$ in place of the uniform draw. Each gene
gets its own seeded start value $z_0 \sim U(0.01, 0.99)$, run for 10 burn-in steps, and
successive iterates give successive wolves. The iterates are rescaled from $(0,1)$ to $[-4, 4]$.

**Fitness.** For a mask $\mathbf{m}$ with $|\mathbf{m}|$ genes selected out of $d$ candidates,

$$
f(\mathbf{m}) = \bigl(1 - \overline{\mathrm{acc}}_{\text{inner}}(\mathbf{m})\bigr) + \lambda \frac{|\mathbf{m}|}{d},
$$

where $\overline{\mathrm{acc}}_{\text{inner}}$ is the mean score of a 3-nearest-neighbour
classifier over a stratified inner cross-validation on the outer training fold. The number of
inner folds is $\min(3, n_{\min})$, with $n_{\min}$ the smallest class size, and falls back to
unstratified folds if a class has a single sample. Balanced accuracy is used when the
largest-to-smallest class ratio exceeds 1.5, and accuracy otherwise. An empty mask has
$f = 1$, and fitness values are cached per unique mask within a run.

**Prefilter.** To make the search tractable on $p > 20{,}000$ genes, the genes are first
ranked by ANOVA F-score on the outer training fold, and the top $P$ are kept. Prefilter
leakage would bias the inner score: if the inner-validation rows helped choose the
candidate genes, the inner CV would score those rows on genes picked partly for fitting them.
On pure-noise data this gives inner accuracies close to 1 (see `test_inner_prefilter_removes_inner_selection_bias`).
Each inner fold therefore recomputes its own top-$P$ list from its inner-training rows, and
evaluates a mask only on the intersection of the mask with that list. The search space is the
union of the outer-fold top-$P$ list and the inner-fold lists ($d \geq P$). With this
correction the initial inner error on noise data is at chance (≈0.5).

**Panel size.** In *free mode* the panel is the best mask found (GWO-free). For comparisons
at equal panel size $k$, GWO's genes are ranked by F-score and the top $k$ are kept. If GWO
selected fewer than $k$, the panel is padded with the next-best prefiltered genes by F-score.

**Hyperparameters.** Values were fixed before running on real data and were not tuned
towards any outcome.

| Parameter | Value | Note |
|---|---|---|
| Wolves $N$ | 20 | `--quick`: 10 |
| Iterations $T$ | 40 | `--quick`: 15 |
| Gene-count penalty $\lambda$ | 0.01 | weight on fraction selected |
| Prefilter $P$ | 2000 | `--quick`: 1000 |
| Inner CV folds | 3 | capped by smallest class |
| Fitness classifier | kNN, $k=3$, Euclidean | linear SVM also implemented |
| Initial position range | $[-4, 4]$ | positions clipped to $[-6, 6]$ |
| Transfer | S (sigmoid) / V ($\lvert\tanh\rvert$) | GWO / GWO-chaotic-V |
| Initialisation | uniform / logistic map | GWO / GWO-chaotic-V |
| Seed | 0 | per GWO run |

**Complexity.** One fitness evaluation costs one kNN fit per inner fold, so
$O(F\, n_{\text{tr}}\, n_{\text{va}}\, d)$ for $F$ inner folds. A run performs at most $N T$
evaluations (fewer when masks repeat), for $O(N T F n^2 d)$ in total with $n$ training samples.
All wolves are scored together as batched matrix products. On a 72-sample, 5000-gene dataset,
one GWO run with $N=10, T=15, P=1000$ took about 0.2 s on a laptop CPU. The
default settings take [[t_default]] s per fold on [[dataset]].

### Evaluation protocol

All results use nested cross-validation (`src/cv_runner.py`). The outer loop is
`RepeatedStratifiedKFold` with 5 folds × 3 repeats and `random_state=42` (folds are capped by
the smallest class size). Fold indices are saved to `results/splits_<dataset>.json` on the
first run and reloaded on every later run, so every method sees identical splits. Within each
outer fold:

1. a `StandardScaler` is fit on the training rows and applied to both training and test rows;
2. each selector receives only the scaled training rows and labels and returns $k$ gene indices;
3. each classifier (linear SVM, $C=1$; 3-NN; random forest with 200 trees, `random_state=0`)
   is trained on the selected genes of the training rows and scored once on the test rows.

The selectors compared were ANOVA F-score, L1-regularised logistic regression (LASSO),
random-forest impurity importance, [[mRMR, if included]], GWO (S-shaped, random init) and
GWO-chaotic-V, at $k \in \{10, 20, 30, 50\}$, plus GWO-free at its own panel size.

**Leakage controls.** No step uses test-fold rows, including scaling, prefiltering, selection
and model fitting. GWO's inner CV adds a second level of separation as described above. Two
automated tests check this end to end. With random labels, the pipeline's mean accuracy is at
chance (0.49 on a 100 × 500 noise matrix). Selecting genes on the full dataset under the same
CV instead gives 0.71, so the test would catch that leak.

**Metrics.** For each outer fold we record accuracy, macro-F1, balanced accuracy, per-class
recall (full vector and its minimum), the number of genes, the selected gene indices (used
for the stability analysis in [[Mitali's section]]) and the selection wall time.

**Statistical reporting.** We report means over the 15 outer folds with 95% Student-t
intervals ($\bar{x} \pm t_{0.975,\,n-1}\, s/\sqrt{n}$). Folds from repeated CV share training
data and are not independent, so these intervals are too narrow and serve as descriptive
summaries. [[Paired comparisons: e.g. Wilcoxon signed-rank across folds / corrected
resampled t-test (Nadeau & Bengio, 2003), if added.]]

## Results

### GWO vs baselines

**Table [[1]].** Mean (95% CI) outer-fold accuracy, macro-F1 and minimum per-class recall
at $k = 30$, by selector and classifier, on [[dataset(s)]].

| Selector | Classifier | Accuracy | Macro-F1 | Min recall | Genes |
|---|---|---|---|---|---|
| ANOVA | SVM / kNN / RF | [[ ]] | [[ ]] | [[ ]] | 30 |
| LASSO | … | [[ ]] | [[ ]] | [[ ]] | 30 |
| RF importance | … | [[ ]] | [[ ]] | [[ ]] | 30 |
| [[mRMR]] | … | [[ ]] | [[ ]] | [[ ]] | 30 |
| GWO | … | [[ ]] | [[ ]] | [[ ]] | 30 |
| GWO-chaotic-V | … | [[ ]] | [[ ]] | [[ ]] | 30 |
| GWO-free | … | [[ ]] | [[ ]] | [[ ]] | [[mean n]] |

Figure [[2]] (`figures/selector_comparison_<dataset>.png`) shows mean macro-F1 with 95% CI
per selector and classifier. [[Describe the ordering of methods, whether GWO's intervals
overlap the baselines', and which classifier benefits most. If GWO does not outperform the
baselines, say so directly; that is the answer to the research question.]]

### Convergence

Figure [[3]] (`figures/convergence_<dataset>.png`) shows the best fitness per iteration,
averaged over outer folds, with ±1 SD bands. [[Report the iteration at which the curves
plateau, the final fitness of each variant, and whether chaotic initialisation with the
V-shaped transfer converges faster or to a different level than the standard variant.]]

### Accuracy vs number of genes

Figure [[4]] (`figures/accuracy_vs_genes_<dataset>.png`) plots mean accuracy (95% CI)
against $k$, with one panel per classifier. [[Describe the trend with k for each selector
and whether any method reaches its best accuracy at a small k.]] GWO-free selected
[[mean ± SD]] genes per fold, at accuracy [[ ]].

### Runtime

[[Mean selection time per fold: ANOVA [[ ]] s, LASSO [[ ]] s, RF [[ ]] s, GWO [[ ]] s,
GWO-chaotic-V [[ ]] s.]]

## Role and responsibilities (Snehal)

Snehal designed and implemented the Binary GWO selector and its chaotic/V-shaped variant,
the nested cross-validation runner with frozen splits and leakage controls, the
convergence, accuracy-vs-genes and selector-comparison figures, the experiment CLI, and the
automated tests for these components. Snehal also wrote the GWO Methods and Results
sections of this report. Data loading, the baseline selectors, the stability analysis and
the GO enrichment analysis were done by Mitali.

## Limitations

- **Small n.** With fewer than 100 samples, each outer test fold holds roughly 10–20
  samples, so one misclassification changes fold accuracy by 5–10 points. Differences
  between selectors smaller than this should not be over-interpreted.
- **High variance.** Accuracy varies widely across folds, and the t-intervals understate
  this because repeated-CV folds are correlated. Inner CV on about 40 training samples gives
  a coarse, noisy fitness signal, so many masks tie and the search has little to separate them.
- **Weak sparsity pressure.** With $\lambda = 0.01$ and a symmetric initial range, free-mode
  GWO kept roughly half the candidate genes in pilot runs. It therefore does not by itself
  yield small panels, and fixed-k panels depend heavily on the final F-score ranking. On
  synthetic data, GWO's top-10 panels contained 3.0 of 5 informative genes on average, versus
  2.0 for a random half-mask with the same ranking step. The search contributes something,
  but modestly.
- **V-shaped transfer on positions.** As specified, the V-shaped rule applies
  $|\tanh(x)|$ to the position itself, not to a velocity or step, so bits keep flipping even
  after the positions converge. This may handicap GWO-chaotic-V relative to formulations that
  apply the V-function to the step.
- **Runtime.** GWO costs orders of magnitude more than filter methods, since every fitness
  evaluation is a full inner CV. With a k-sweep, GWO is run once per fold and reused for every
  k.
- **Prefilter dependence.** GWO only searches genes that pass the ANOVA prefilter, so a gene
  that matters only together with others but has a weak marginal F-score cannot be selected.
  Results depend on $P$, and fixed-k panels inherit the F-score ranking.
- **Single fitness classifier.** The fitness uses 3-NN, so panels may favour kNN-friendly
  structure. This may advantage kNN in the outer evaluation relative to SVM and RF.
