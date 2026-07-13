"""第12回課題用: Worker数・個体数・処理時間・最良解の比較実験。"""

from __future__ import annotations

import csv
import multiprocessing
import time
from pathlib import Path

import numpy as np
from numpy.random import default_rng


def rastrigin(x: np.ndarray) -> float:
    return (
        20.0
        + x[0] ** 2
        + x[1] ** 2
        - 10.0 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))
    )


class GeneticAlgorithm:
    def __init__(self, pop_size: int, workers: int, seed: int):
        self.pop_size = pop_size
        self.workers = workers
        self.rng = default_rng(seed)
        self.bounds = [-5.12, 5.12]
        self.mutation_rate = 0.15
        self.mutation_std = 0.3
        self.crossover_rate = 0.8
        self.tournament_size = 3

    def evaluate(self, population: np.ndarray, parallel: bool) -> np.ndarray:
        if not parallel:
            return np.array([rastrigin(x) for x in population])

        # 配布Notebookと同じく、評価のたびにPoolを作成する。
        context = multiprocessing.get_context("fork")
        with context.Pool(processes=self.workers) as pool:
            return np.array(pool.map(rastrigin, population))

    def tournament_select(
        self, population: np.ndarray, fitnesses: np.ndarray
    ) -> np.ndarray:
        indices = self.rng.choice(
            len(population), size=(len(population), self.tournament_size)
        )
        tournament_fitnesses = fitnesses[indices]
        winners = indices[
            np.arange(len(population)), np.argmin(tournament_fitnesses, axis=1)
        ]
        return population[winners]

    def crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        alpha = self.rng.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        mutated = individual.copy()
        for dim in range(len(mutated)):
            if self.rng.random() < self.mutation_rate:
                mutated[dim] += self.rng.normal(0, self.mutation_std)
                mutated[dim] = np.clip(mutated[dim], *self.bounds)
        return mutated

    def run(self, generations: int = 80, parallel: bool = True) -> float:
        population = self.rng.uniform(
            self.bounds[0], self.bounds[1], (self.pop_size, 2)
        )

        for _ in range(generations):
            fitnesses = self.evaluate(population, parallel)
            best_individual = population[np.argmin(fitnesses)].copy()
            parents = self.tournament_select(population, fitnesses)

            offspring = []
            index = 0
            while len(offspring) < self.pop_size:
                parent1 = parents[index % len(parents)]
                parent2 = parents[(index + 1) % len(parents)]
                child1, child2 = self.crossover(parent1, parent2)
                offspring.append(self.mutate(child1))
                if len(offspring) < self.pop_size:
                    offspring.append(self.mutate(child2))
                index += 2

            population = np.array(offspring)
            offspring_fitnesses = self.evaluate(population, parallel)
            population[np.argmax(offspring_fitnesses)] = best_individual

        final_fitnesses = self.evaluate(population, parallel)
        return float(np.min(final_fitnesses))


def run_timing_experiment(output_dir: Path) -> None:
    rows = []
    for pop_size in [25, 50, 100, 200]:
        for workers in [1, 2, 4, 8]:
            ga = GeneticAlgorithm(pop_size=pop_size, workers=workers, seed=42)
            start = time.perf_counter()
            best_fitness = ga.run(generations=80, parallel=True)
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "population": pop_size,
                    "workers": workers,
                    "time_s": f"{elapsed:.3f}",
                    "best_fitness": f"{best_fitness:.10f}",
                }
            )
            print(
                f"population={pop_size:3d}, workers={workers}, "
                f"time={elapsed:.3f}s, best={best_fitness:.10f}",
                flush=True,
            )

    with (output_dir / "ga_worker_population_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_quality_experiment(output_dir: Path) -> None:
    rows = []
    for pop_size in [10, 25, 50, 100, 200]:
        results = []
        for seed in range(10):
            ga = GeneticAlgorithm(pop_size=pop_size, workers=1, seed=seed)
            results.append(ga.run(generations=80, parallel=False))

        rows.append(
            {
                "population": pop_size,
                "median_best": f"{np.median(results):.10f}",
                "mean_best": f"{np.mean(results):.10f}",
                "best_run": f"{np.min(results):.10f}",
                "worst_run": f"{np.max(results):.10f}",
            }
        )
        print(
            f"population={pop_size:3d}, median={np.median(results):.10f}, "
            f"worst={np.max(results):.10f}",
            flush=True,
        )

    with (output_dir / "ga_population_quality_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    run_timing_experiment(output_dir)
    run_quality_experiment(output_dir)


if __name__ == "__main__":
    main()
