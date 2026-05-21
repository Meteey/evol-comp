from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from evaluation import evaluate_model
from ga_feature_selection import GeneticFeatureSelector
from plotting import (
    plot_confusion_matrix,
    plot_f1_comparison,
    plot_feature_count_comparison,
    plot_fitness_curve,
)
from statistical_tests import run_statistical_tests

N_SEEDS = 30
SPLIT_SEED = 42
TEST_SIZE = 0.2
RESULTS_DIR = Path(__file__).parent / "results"

GA_FIT_N_ESTIMATORS = 20

GA_BASE_SETTINGS = dict(
    population_size=20,
    generations=20,
    crossover_rate=0.8,
    mutation_rate=0.03,
    elitism_count=2,
    tournament_size=3,
    feature_penalty_weight=0.02,
    cv_folds=5,
    scoring="f1",
)

_GA_CSV_COLS = [
    "seed", "selected_feature_indices", "selected_feature_count",
    "selected_feature_ratio", "accuracy", "precision", "recall",
    "f1", "roc_auc", "runtime_seconds", "best_fitness", "final_generation",
]

_RAND_CSV_COLS = [
    "seed", "selected_feature_indices", "selected_feature_count",
    "selected_feature_ratio", "accuracy", "precision", "recall",
    "f1", "roc_auc", "runtime_seconds",
]

_METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def classifier_factory_ga() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=GA_FIT_N_ESTIMATORS, random_state=42, n_jobs=1)


def classifier_factory() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


def _run_ga(seed: int, X_train, X_test, y_train, y_test, n_features: int) -> dict:
    settings = {**GA_BASE_SETTINGS, "random_state": seed}
    ga = GeneticFeatureSelector(classifier_factory=classifier_factory_ga, **settings)

    t0 = time.perf_counter()
    _, best_fit, selected_idx, history = ga.fit(X_train, y_train, verbose=False)
    runtime = time.perf_counter() - t0

    metrics = evaluate_model(
        X_train, X_test, y_train, y_test,
        selected_idx, classifier_factory, f"ga_seed_{seed}",
    )

    return {
        "seed": seed,
        "selected_feature_indices": str(list(int(i) for i in selected_idx)),
        "selected_feature_count": int(len(selected_idx)),
        "selected_feature_ratio": round(len(selected_idx) / n_features, 4),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "runtime_seconds": round(runtime, 2),
        "best_fitness": round(best_fit, 6),
        "final_generation": int(ga.best_generation_),
        "_history": history,
        "_confusion_matrix": metrics["confusion_matrix"],
        "_selected_idx": selected_idx,
    }


def _run_random(seed: int, k: int, X_train, X_test, y_train, y_test, n_features: int) -> dict:
    rng = np.random.default_rng(seed)
    selected_idx = np.sort(rng.choice(n_features, size=k, replace=False))

    t0 = time.perf_counter()
    metrics = evaluate_model(
        X_train, X_test, y_train, y_test,
        selected_idx, classifier_factory, f"random_seed_{seed}",
    )
    runtime = time.perf_counter() - t0

    return {
        "seed": seed,
        "selected_feature_indices": str(list(int(i) for i in selected_idx)),
        "selected_feature_count": int(k),
        "selected_feature_ratio": round(k / n_features, 4),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "runtime_seconds": round(runtime, 6),
    }


def _mean_std_row(df: pd.DataFrame, method: str) -> dict:
    row: dict = {"method": method}
    for col in _METRIC_COLS + ["selected_feature_count", "runtime_seconds"]:
        row[f"{col}_mean"] = round(float(df[col].mean()), 6)
        row[f"{col}_std"] = round(float(df[col].std(ddof=1)), 6)
    return row


def _all_features_row(metrics: dict, n_features: int) -> dict:
    row: dict = {"method": "All Features"}
    for col in _METRIC_COLS:
        row[f"{col}_mean"] = round(float(metrics[col]), 6)
        row[f"{col}_std"] = 0.0
    row["selected_feature_count_mean"] = float(n_features)
    row["selected_feature_count_std"] = 0.0
    row["runtime_seconds_mean"] = None
    row["runtime_seconds_std"] = None
    return row


def main() -> None:
    os.environ["PYTHONHASHSEED"] = str(SPLIT_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    class_names = list(data.target_names)
    n_features = X.shape[1]

    print("=" * 65)
    print("Genetic Algorithm Based Feature Selection — Repeated Experiment")
    print(f"Dataset : Breast Cancer Wisconsin (sklearn)")
    print(f"Samples : {X.shape[0]}   Features: {n_features}")
    print(f"Classes : {class_names}")
    print("=" * 65)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED
    )
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    print(f"\n[1/4] Running {N_SEEDS} GA experiments (seeds 0-{N_SEEDS - 1}) ...")
    ga_results: list[dict] = []
    for seed in range(N_SEEDS):
        result = _run_ga(seed, X_train, X_test, y_train, y_test, n_features)
        ga_results.append(result)
        print(
            f"  seed={seed:>2}  f1={result['f1']:.4f}  "
            f"k={result['selected_feature_count']:>2}  "
            f"t={result['runtime_seconds']:.1f}s  "
            f"best_gen={result['final_generation']}"
        )

    ga_df = pd.DataFrame([{c: r[c] for c in _GA_CSV_COLS} for r in ga_results])
    ga_df.to_csv(RESULTS_DIR / "ga_repeated_runs.csv", index=False)
    print(
        f"\n  => ga_repeated_runs.csv  "
        f"F1 mean={ga_df['f1'].mean():.4f}  std={ga_df['f1'].std(ddof=1):.4f}  "
        f"k mean={ga_df['selected_feature_count'].mean():.1f}"
    )

    print(f"\n[2/4] Running {N_SEEDS} random baseline experiments ...")
    random_results: list[dict] = []
    for seed in range(N_SEEDS):
        k = ga_results[seed]["selected_feature_count"]
        result = _run_random(seed, k, X_train, X_test, y_train, y_test, n_features)
        random_results.append(result)
        print(f"  seed={seed:>2}  f1={result['f1']:.4f}  k={k:>2}")

    random_df = pd.DataFrame([{c: r[c] for c in _RAND_CSV_COLS} for r in random_results])
    random_df.to_csv(RESULTS_DIR / "random_repeated_runs.csv", index=False)
    print(
        f"\n  => random_repeated_runs.csv  "
        f"F1 mean={random_df['f1'].mean():.4f}  std={random_df['f1'].std(ddof=1):.4f}"
    )

    print("\n[3/4] Running all-features baseline ...")
    all_idx = np.arange(n_features)
    all_metrics = evaluate_model(
        X_train, X_test, y_train, y_test,
        all_idx, classifier_factory, "all_features",
    )
    all_features_df = pd.DataFrame([
        {
            "method": "all_features",
            "selected_feature_count": n_features,
            "selected_feature_ratio": 1.0,
            **{col: all_metrics[col] for col in _METRIC_COLS},
        }
    ])
    all_features_df.to_csv(RESULTS_DIR / "all_features_baseline.csv", index=False)
    print(f"  => all_features_baseline.csv  F1={all_metrics['f1']:.4f}")

    print("\n[4/4] Statistical significance test (Mann-Whitney U) ...")
    tests_df = run_statistical_tests(ga_df, random_df)
    tests_df.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)
    row = tests_df.iloc[0]
    print(
        f"  Mann-Whitney U:  statistic={row['statistic']:.2f}  "
        f"p={row['p_value']:.6f}  "
        f"significant={'YES' if row['significant_at_0_05'] else 'NO'}"
    )

    summary_df = pd.DataFrame([
        _mean_std_row(ga_df, "GA"),
        _mean_std_row(random_df, "Random Subset"),
        _all_features_row(all_metrics, n_features),
    ])
    summary_df.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    best_idx = int(ga_df["f1"].idxmax())
    best_seed = int(ga_df.loc[best_idx, "seed"])
    best_ga = ga_results[best_idx]
    print(f"\nBest GA run: seed={best_seed}  f1={ga_df.loc[best_idx, 'f1']:.4f}")

    plot_fitness_curve(
        best_ga["_history"],
        RESULTS_DIR / "fitness_curve_best_ga.png",
    )
    plot_f1_comparison(
        ga_df["f1"],
        random_df["f1"],
        all_metrics["f1"],
        RESULTS_DIR / "f1_comparison_mean_std.png",
    )
    plot_feature_count_comparison(
        ga_df["selected_feature_count"],
        random_df["selected_feature_count"],
        RESULTS_DIR / "selected_feature_count_comparison.png",
    )
    plot_confusion_matrix(
        best_ga["_confusion_matrix"],
        class_names,
        title=f"Confusion Matrix — Best GA (seed={best_seed})",
        out_path=RESULTS_DIR / "confusion_matrix_best_ga.png",
    )

    print("\n" + "=" * 65)
    print("Summary Table (mean ± std across 30 runs)")
    print("=" * 65)
    display_cols = ["method", "f1_mean", "f1_std", "selected_feature_count_mean", "runtime_seconds_mean"]
    print(summary_df[display_cols].to_string(index=False))

    print("\nResults written to:", RESULTS_DIR)
    print("\nGenerated files:")
    for f in sorted(RESULTS_DIR.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
