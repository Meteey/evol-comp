from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def plot_fitness_curve(history: pd.DataFrame, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["generation"], history["best_fitness"], label="Best fitness", linewidth=2)
    ax.plot(history["generation"], history["avg_fitness"], label="Average fitness", linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("GA Fitness over Generations")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_selected_features_bar(
    selected_indices: Iterable[int],
    feature_names: Iterable[str],
    out_path: str | Path,
) -> None:
    feature_names = list(feature_names)
    selected_indices = list(selected_indices)
    selected_names = [feature_names[i] for i in selected_indices]

    fig, ax = plt.subplots(figsize=(10, max(4, len(selected_names) * 0.35)))
    y_pos = np.arange(len(selected_names))
    ax.barh(y_pos, np.ones(len(selected_names)), color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(selected_names)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_title(f"GA-Selected Features ({len(selected_names)} total)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_f1_comparison(
    ga_f1: ArrayLike,
    random_f1: ArrayLike,
    all_features_f1: float,
    out_path: str | Path,
) -> None:
    ga_f1 = np.asarray(ga_f1)
    random_f1 = np.asarray(random_f1)

    methods = ["GA", "Random Subset", "All Features"]
    means = [float(ga_f1.mean()), float(random_f1.mean()), all_features_f1]
    stds = [float(ga_f1.std(ddof=1)), float(random_f1.std(ddof=1)), 0.0]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["steelblue", "tomato", "seagreen"]
    bars = ax.bar(
        methods, means, yerr=stds, capsize=7,
        color=colors, alpha=0.85, width=0.5,
        error_kw=dict(linewidth=1.8, ecolor="black"),
    )
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) + 0.003,
            f"{mean:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Comparison (30 repeated runs, mean ± std)")
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_count_comparison(
    ga_counts: ArrayLike,
    random_counts: ArrayLike,
    out_path: str | Path,
) -> None:
    ga_counts = np.asarray(ga_counts, dtype=float)
    random_counts = np.asarray(random_counts, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))

    data = [ga_counts, random_counts]
    bp = ax.boxplot(
        data, labels=["GA", "Random Subset"],
        patch_artist=True, widths=0.4,
        medianprops=dict(color="black", linewidth=2),
    )
    colors = ["steelblue", "tomato"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    rng = np.random.default_rng(0)
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=vals.shape)
        ax.scatter(i + jitter, vals, color=color, alpha=0.7, s=25, zorder=3)

    ax.set_ylabel("Number of Selected Features")
    ax.set_title("Selected Feature Count per Run (30 seeds)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Iterable[str],
    title: str,
    out_path: str | Path,
) -> None:
    """Plot a 2x2 confusion matrix with annotations."""
    class_names = list(class_names)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
