# Genetic Algorithm Based Feature Selection for Breast Cancer Classification

A from-scratch Genetic Algorithm (no DEAP / PyGAD) that selects an optimal
subset of features from the Breast Cancer Wisconsin dataset, then compares
the chosen subset against two baselines using a Random Forest classifier.

## What it does

1. Loads `sklearn.datasets.load_breast_cancer` (569 samples, 30 features).
2. Splits data stratified into 80% train / 20% test.
3. Runs a Genetic Algorithm on the **training data only** to evolve a
   binary mask over the 30 features.
4. Trains a `RandomForestClassifier` on the test set using:
   - all 30 features,
   - a random subset of equal size to the GA selection,
   - the GA-selected subset.
5. Saves metrics, plots, and the per-generation history to `results/`.

## How the GA works (short)

- **Chromosome**: binary vector of length 30 (1 = use this feature).
- **Fitness**: `f1_score` from 5-fold stratified CV on the training set
  minus a small penalty for the number of selected features:

  ```
  fitness = mean_CV_f1 - 0.02 * (k / 30)
  ```

  Zero-feature chromosomes are forbidden (penalised heavily).
- **Selection**: tournament selection (size 3).
- **Crossover**: uniform crossover, rate 0.8.
- **Mutation**: bit-flip mutation, per-gene rate 0.03.
- **Elitism**: top 2 chromosomes carried unchanged each generation.
- **Population** = 50, **generations** = 50, seed = 42.

## How to run

```bash
pip install -r requirements.txt
python main.py
```

Runtime is roughly 1–3 minutes on a normal laptop (depends on cores).

## Expected outputs

After running, `results/` will contain:

```
results/
├── metrics_summary.csv          # accuracy/precision/recall/f1/roc_auc for the 3 methods
├── selected_features.csv        # indices + names of GA-selected features
├── generation_history.csv       # per-generation best/avg fitness etc.
├── fitness_curve.png            # best & avg fitness vs generation
├── selected_features_bar.png    # bar chart of selected feature names
├── confusion_matrix_all_features.png
├── confusion_matrix_random_subset.png
└── confusion_matrix_ga_subset.png
```

## Experimental Protocol

| Step | Detail |
|------|--------|
| **Dataset** | Breast Cancer Wisconsin (sklearn built-in), 569 samples × 30 features, binary classification |
| **Train/test split** | Stratified 80/20 (455 train, 114 test), fixed seed |
| **GA fitness** | 5-fold stratified cross-validation F1 on training data only (no test-set leakage) |
| **Repeated GA runs** | 30 independent runs with seeds 0–29; each run: 50 chromosomes × 50 generations |
| **Repeated random baseline** | 30 independent runs with seeds 0–29; each run randomly selects the same number of features as the paired GA run |
| **All-features baseline** | Single deterministic run using all 30 features |
| **Statistical test** | Mann-Whitney U test (`scipy.stats.mannwhitneyu`, `alternative='greater'`): GA F1 > Random F1; α = 0.05 |
| **Metrics** | Accuracy, Precision, Recall, F1, ROC-AUC (all measured on held-out test set) |

### Output files

```
results/
├── ga_repeated_runs.csv              # 30 GA runs (seed, features, metrics, runtime, best_fitness, final_generation)
├── random_repeated_runs.csv          # 30 random baseline runs
├── all_features_baseline.csv         # single all-features evaluation
├── statistical_tests.csv             # Mann-Whitney U result
├── summary_table.csv                 # mean ± std for all three methods
├── fitness_curve_best_ga.png         # best & avg fitness for the top GA run
├── f1_comparison_mean_std.png        # bar chart mean ± std F1 for all methods
├── selected_feature_count_comparison.png
└── confusion_matrix_best_ga.png
```

## Demo

Run `python main.py` and watch:

- the header (dataset name, samples, features, GA settings),
- per-generation progress lines,
- the final selected feature list,
- the comparison table for the three methods.

The expected story: the GA reaches accuracy/F1 comparable to (or slightly
better than) using all 30 features, while using about half of them,
clearly beating a random subset of the same size.
