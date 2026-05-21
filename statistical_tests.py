from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu


def run_statistical_tests(ga_df: pd.DataFrame, random_df: pd.DataFrame) -> pd.DataFrame:
    ga_f1 = ga_df["f1"].values
    random_f1 = random_df["f1"].values

    stat, p = mannwhitneyu(ga_f1, random_f1, alternative="greater")

    results = [
        {
            "test_name": "Mann-Whitney U",
            "metric": "f1",
            "method_a": "GA",
            "method_b": "Random Subset",
            "statistic": float(stat),
            "p_value": float(p),
            "significant_at_0_05": bool(p < 0.05),
            "ga_mean": float(ga_f1.mean()),
            "ga_std": float(ga_f1.std(ddof=1)),
            "random_mean": float(random_f1.mean()),
            "random_std": float(random_f1.std(ddof=1)),
        }
    ]

    return pd.DataFrame(results)


if __name__ == "__main__":
    results_dir = Path(__file__).parent / "results"
    ga_df = pd.read_csv(results_dir / "ga_repeated_runs.csv")
    random_df = pd.read_csv(results_dir / "random_repeated_runs.csv")
    tests_df = run_statistical_tests(ga_df, random_df)
    tests_df.to_csv(results_dir / "statistical_tests.csv", index=False)
    print(tests_df.to_string(index=False))
