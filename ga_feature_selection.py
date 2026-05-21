from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def _default_classifier_factory() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=30,
        random_state=42,
        n_jobs=1,
    )


class GeneticFeatureSelector:
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.03,
        elitism_count: int = 2,
        tournament_size: int = 3,
        feature_penalty_weight: float = 0.02,
        cv_folds: int = 5,
        scoring: str = "f1",
        random_state: int = 42,
        classifier_factory: Optional[Callable[[], RandomForestClassifier]] = None,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.feature_penalty_weight = feature_penalty_weight
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.classifier_factory = classifier_factory or _default_classifier_factory

        self._rng = np.random.default_rng(random_state)

        self.n_features_: int = 0
        self.best_chromosome_: Optional[np.ndarray] = None
        self.best_fitness_: float = -np.inf
        self.best_cv_score_: float = -np.inf
        self.best_generation_: int = 0
        self.history_: Optional[pd.DataFrame] = None

        # key: chromosome.tobytes()
        # value: (fitness, raw_cv_score)
        self._fitness_cache: dict[bytes, Tuple[float, float]] = {}

    def _init_population(self, n_features: int) -> np.ndarray:
        pop = self._rng.integers(0, 2, size=(self.population_size, n_features))

        for i in range(self.population_size):
            if pop[i].sum() == 0:
                pop[i, self._rng.integers(0, n_features)] = 1

        return pop.astype(np.uint8)

    def _fitness(
        self,
        chromosome: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float]:
        chromosome = np.asarray(chromosome, dtype=np.uint8)
        key = chromosome.tobytes()

        cached = self._fitness_cache.get(key)
        if cached is not None:
            return cached

        selected = np.where(chromosome == 1)[0]
        n_features = chromosome.shape[0]

        if selected.size == 0:
            result = (-1.0, 0.0)
            self._fitness_cache[key] = result
            return result

        X_sub = X[:, selected]

        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        clf = self.classifier_factory()

        scores = cross_val_score(
            clf,
            X_sub,
            y,
            cv=cv,
            scoring=self.scoring,
            n_jobs=None,
        )

        cv_score = float(np.mean(scores))
        penalty = self.feature_penalty_weight * (selected.size / n_features)
        fitness = cv_score - penalty

        result = (fitness, cv_score)
        self._fitness_cache[key] = result
        return result

    def _tournament_select(
        self,
        population: np.ndarray,
        fitnesses: np.ndarray,
    ) -> np.ndarray:
        idx = self._rng.integers(
            0,
            population.shape[0],
            size=self.tournament_size,
        )
        best = idx[np.argmax(fitnesses[idx])]
        return population[best].copy()

    def _uniform_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        mask = self._rng.integers(0, 2, size=parent1.shape).astype(bool)

        child1 = np.where(mask, parent1, parent2).astype(np.uint8)
        child2 = np.where(mask, parent2, parent1).astype(np.uint8)

        return child1, child2

    def _bitflip_mutate(self, chromosome: np.ndarray) -> np.ndarray:
        flips = self._rng.random(chromosome.shape) < self.mutation_rate
        mutated = np.where(flips, 1 - chromosome, chromosome).astype(np.uint8)

        if mutated.sum() == 0:
            mutated[self._rng.integers(0, mutated.shape[0])] = 1

        return mutated

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float, np.ndarray, pd.DataFrame]:

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        self.n_features_ = X_train.shape[1]

        # Reset state for this run.
        self._rng = np.random.default_rng(self.random_state)
        self._fitness_cache = {}

        self.best_chromosome_ = None
        self.best_fitness_ = -np.inf
        self.best_cv_score_ = -np.inf
        self.best_generation_ = 0
        self.history_ = None

        population = self._init_population(self.n_features_)

        fitnesses = np.zeros(self.population_size, dtype=float)
        cv_scores = np.zeros(self.population_size, dtype=float)

        for i in range(self.population_size):
            fitnesses[i], cv_scores[i] = self._fitness(
                population[i],
                X_train,
                y_train,
            )

        history_rows: List[dict] = []

        for gen in range(self.generations):
            best_idx = int(np.argmax(fitnesses))
            best_fit = float(fitnesses[best_idx])
            avg_fit = float(np.mean(fitnesses))
            best_cv = float(cv_scores[best_idx])
            avg_selected = float(np.mean(population.sum(axis=1)))
            best_selected = int(population[best_idx].sum())

            history_rows.append(
                {
                    "generation": gen,
                    "best_fitness": best_fit,
                    "avg_fitness": avg_fit,
                    "best_cv_score": best_cv,
                    "avg_selected_features": avg_selected,
                    "best_selected_features": best_selected,
                    "fitness_cache_size": len(self._fitness_cache),
                }
            )

            if best_fit > self.best_fitness_:
                self.best_fitness_ = best_fit
                self.best_chromosome_ = population[best_idx].copy()
                self.best_cv_score_ = best_cv
                self.best_generation_ = gen

            if verbose:
                print(
                    f"  Gen {gen + 1:>3}/{self.generations} | "
                    f"best_fit={best_fit:.4f} | "
                    f"avg_fit={avg_fit:.4f} | "
                    f"best_cv={best_cv:.4f} | "
                    f"best_k={best_selected} | "
                    f"avg_k={avg_selected:.1f} | "
                    f"cache={len(self._fitness_cache)}"
                )

            elite_idx = np.argsort(fitnesses)[::-1][: self.elitism_count]
            new_population = [population[i].copy() for i in elite_idx]

            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population, fitnesses)
                parent2 = self._tournament_select(population, fitnesses)

                child1, child2 = self._uniform_crossover(parent1, parent2)

                child1 = self._bitflip_mutate(child1)
                child2 = self._bitflip_mutate(child2)

                new_population.append(child1)

                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = np.asarray(new_population, dtype=np.uint8)

            for i in range(self.population_size):
                fitnesses[i], cv_scores[i] = self._fitness(
                    population[i],
                    X_train,
                    y_train,
                )

        best_idx = int(np.argmax(fitnesses))

        if float(fitnesses[best_idx]) > self.best_fitness_:
            self.best_fitness_ = float(fitnesses[best_idx])
            self.best_chromosome_ = population[best_idx].copy()
            self.best_cv_score_ = float(cv_scores[best_idx])
            self.best_generation_ = self.generations

        if self.best_chromosome_ is None:
            raise RuntimeError("GA finished without selecting a best chromosome.")

        self.history_ = pd.DataFrame(history_rows)
        selected_indices = np.where(self.best_chromosome_ == 1)[0]

        return (
            self.best_chromosome_,
            self.best_fitness_,
            selected_indices,
            self.history_,
        )