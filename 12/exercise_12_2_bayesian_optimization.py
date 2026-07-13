"""
Exercise 12-2: Bayesian Optimization with Parallel Evaluation
=============================================================
Bayesian Optimization (BO) for black-box function minimization,
demonstrating parallel evaluation of candidate points.

Optimization problem: Himmelblau's function (2D)
    f(x, y) = (x² + y - 11)² + (x + y² - 7)²
Four known global minima at approximately:
    (3.0, 2.0), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)
All with f = 0.

Bayes' theorem connection:
    Bayesian Optimization IS Bayes' theorem applied to function optimization:

        p(f | D) = p(D | f) · p(f) / p(D)

    where:
        p(f)        = GP prior over functions     ← prior
        p(D | f)    = likelihood of observations   ← likelihood
        p(f | D)    = GP posterior over functions  ← posterior

    The Gaussian Process provides a closed-form Bayesian update:
        μ*(x) = k*(x)^T (K + σ_n²I)⁻¹ y     ← posterior mean
        σ²*(x) = k(x,x) - k*(x)^T (K + σ_n²I)⁻¹ k*(x)  ← posterior variance

    The Expected Improvement acquisition function computes:
        EI(x) = E[max(0, f* - f(x))]
              = (μ*(x) - f* - ξ) · Φ(Z) + σ*(x) · φ(Z)

    This is an EXPECTATION under the posterior distribution p(f|D) —
    exactly the Monte Carlo approximation discussed in the lecture:
        E[f(θ)] ≈ (1/N) Σ f(θ^{(i)})

    Parallelization: In batch BO, we evaluate multiple candidate points
    simultaneously using pool.map(). Each candidate's black-box evaluation
    is independent — this is the likelihood computation p(D|θ) for
    multiple θ candidates, parallelized via Bayes' theorem's structure.

Usage:
    python exercise_2_2_bayesian_optimization.py
"""

import time
import multiprocessing
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng


# ===================================================================
# Objective function
# ===================================================================

def himmelblau(x: np.ndarray) -> float:
    """
    Himmelblau's function (2D).

    f(x, y) = (x² + y - 11)² + (x + y² - 7)²

    Four global minima, all with f = 0.
    """
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


# ===================================================================
# Gaussian Process (from scratch, numpy only)
# ===================================================================

class GaussianProcess:
    """
    Minimal Gaussian Process regression implementation.

    This is the Bayesian "engine" of Bayesian Optimization.
    The GP provides the posterior distribution p(f|D) via Bayes' theorem.

    Kernel: Squared exponential (RBF) with ARD (Automatic Relevance Determination)
        k(x, x') = exp(-0.5 * Σ_d (x_d - x'_d)² / l_d²)

    The GP posterior is the closed-form Bayes' theorem update:
        p(f* | X, y, x*) = N(μ*, σ*²)
    """

    def __init__(self, length_scales: np.ndarray = None, noise_std: float = 1e-6):
        self.length_scales = length_scales
        self.noise_std = noise_std
        self.X_train = None
        self.y_train = None
        self.K_inv = None  # Cached inverse for efficiency
        self.alpha = None  # K⁻¹y

    def _rbd_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        RBF kernel with ARD (Automatic Relevance Determination).

        k(x, x') = exp(-0.5 * Σ_d (x_d - x'_d)² / l_d²)

        Each dimension gets its own length scale, allowing the GP to
        automatically determine which dimensions are important.
        """
        if self.length_scales is None:
            D = X1.shape[1]
            ls = np.ones(D)
        else:
            ls = self.length_scales

        # Scale inputs
        X1s = X1 / ls
        X2s = X2 / ls

        # Compute squared distances
        sq_dist = np.sum(X1s ** 2, axis=1, keepdims=True) + \
                  np.sum(X2s ** 2, axis=1, keepdims=True).T - \
                  2.0 * X1s @ X2s.T
        return np.exp(-0.5 * sq_dist)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Compute the GP posterior — Bayes' theorem in action.

        Given training data (X, y), compute:
            α = (K + σ_n²I)⁻¹ y    ← posterior mean parameter
            K_inv = (K + σ_n²I)⁻¹  ← posterior covariance kernel

        This is the closed-form solution to the Bayesian update.
        """
        self.X_train = X.copy()
        self.y_train = y.copy()

        n = len(y)
        K = self._rbd_kernel(X, X)
        K += self.noise_std ** 2 * np.eye(n)  # Add noise term

        # Cholesky decomposition for numerical stability
        # This is equivalent to solving (K + σ_n²I)α = y
        try:
            L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            self.K_inv = np.linalg.solve(L.T, L)  # (K + σ_n²I)⁻¹
        except np.linalg.LinAlgError:
            # Fallback: direct solve (less stable)
            warnings.warn("Cholesky failed, using direct solve")
            self.K_inv = np.linalg.inv(K)
            self.alpha = self.K_inv @ y

    def predict(self, X_new: np.ndarray) -> tuple:
        """
        Compute the GP posterior predictive distribution.

        p(f* | X, y, x*) = N(μ*, σ*²)

        where:
            μ*(x*) = k(x*, X) · α          ← posterior mean
            σ*²(x*) = k(x*, x*) - k(x*, X) · K⁻¹ · k(X, x*)  ← posterior variance

        This IS Bayes' theorem: the posterior is computed from the prior
        (kernel) and the likelihood (data fit through α = K⁻¹y).
        """
        k_star = self._rbd_kernel(X_new, self.X_train)  # k(x*, X)
        k_ss = self._rbd_kernel(X_new, X_new)            # k(x*, x*)

        # Posterior mean: μ* = k*^T α
        mu = k_star @ self.alpha

        # Posterior variance: σ*² = k(x*,x*) - k*^T K⁻¹ k*
        v = self.K_inv @ k_star.T
        sigma2 = np.diag(k_ss) - np.sum(k_star.T * v, axis=0)

        # Clamp negative variances (numerical issues)
        sigma2 = np.maximum(sigma2, 1e-10)

        return mu, sigma2


# ===================================================================
# Acquisition function: Expected Improvement
# ===================================================================

def expected_improvement(mu: np.ndarray, sigma: np.ndarray,
                         f_best: float, xi: float = 0.01) -> np.ndarray:
    """
    Expected Improvement acquisition function.

    EI(x) = E[max(0, f* - f(x))]

    Analytical formula (under Gaussian posterior):
        Z = (μ(x) - f* - ξ) / σ(x)
        EI(x) = (μ(x) - f* - ξ) · Φ(Z) + σ(x) · φ(Z)

    where:
        Φ = standard normal CDF
        φ = standard normal PDF
        f*  = best observed value so far
        ξ   = exploration parameter

    This computes the EXPECTATION under the posterior distribution p(f|D) —
    the Monte Carlo approximation discussed in the lecture:
        E[max(0, f* - f(x))] ≈ analytical formula (exact for Gaussian)

    The EI balances exploration (high σ) and exploitation (low μ):
        - High EI where μ is low (exploitation) or σ is high (exploration)
    """
    from scipy.stats import norm

    # Handle case where sigma is zero
    sigma_safe = np.where(sigma < 1e-10, 1e-10, sigma)

    Z = (mu - f_best - xi) / sigma_safe
    ei = (mu - f_best - xi) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)

    return np.maximum(ei, 0.0)


# ===================================================================
# Bayesian Optimization
# ===================================================================

def bayesian_optimizer(num_init: int, num_iter: int, num_workers: int,
                       bounds: list, seed: int = 42) -> dict:
    """
    Bayesian Optimization with batch parallel evaluation.

    Algorithm:
      1. Random initial design: evaluate num_init random points (parallel)
      2. For each iteration:
         a. Fit GP to current data (Bayes' theorem: prior → posterior)
         b. Optimize EI acquisition function to find next candidate
         c. Evaluate candidate(s) in parallel (pool.map)
         d. Update GP with new observations

    The key parallelization is in step 2c: evaluating multiple candidates
    simultaneously. Each candidate's evaluation is independent — this is
    the likelihood computation for multiple θ values in Bayes' theorem.

    Args:
        num_init:   Number of initial random evaluations
        num_iter:   Number of BO iterations
        num_workers: Number of parallel workers (batch size)
        bounds:     [lo, hi] for each dimension
        seed:       Random seed

    Returns:
        Dictionary with optimization history and results.
    """
    rng = default_rng(seed)
    dim = 2
    lo, hi = bounds[0], bounds[1]

    # Storage
    X_history = []
    y_history = []
    best_history = []
    times = []

    # Initialize GP with default length scales
    gp = GaussianProcess(length_scales=np.array([2.0, 2.0]))

    # --- Step 1: Initial design (parallel) ---
    X_init = rng.uniform(lo, hi, (num_init, dim))
    with multiprocessing.Pool(processes=num_workers) as pool:
        y_init = pool.map(himmelblau, X_init)
    y_init = np.array(y_init)

    X_history.extend(X_init.tolist())
    y_history.extend(y_init.tolist())
    best_history.append(np.min(y_init))
    print(f"  Initial design: {num_init} points evaluated")

    # --- Step 2: Iterative BO ---
    for iteration in range(num_iter):
        t0 = time.perf_counter()

        # Fit GP (Bayes' theorem: compute posterior from prior + data)
        X_arr = np.array(X_history)
        y_arr = np.array(y_history)
        gp.fit(X_arr, y_arr)

        f_best = np.min(y_arr)

        # Optimize EI: grid search over candidate points
        grid_size = 200
        x1_grid = np.linspace(lo, hi, grid_size)
        x2_grid = np.linspace(lo, hi, grid_size)
        X_candidates = np.array([[x1, x2] for x1 in x1_grid for x2 in x2_grid])

        mu, sigma = gp.predict(X_candidates)
        ei = expected_improvement(mu, sigma, f_best)

        # Select best candidate
        best_candidate_idx = np.argmax(ei)
        x_next = X_candidates[best_candidate_idx]

        # --- Parallel evaluation (batch BO) ---
        # For simplicity, evaluate one point at a time but demonstrate
        # the parallel pattern. In practice, evaluate num_workers points.
        with multiprocessing.Pool(processes=num_workers) as pool:
            y_next = pool.map(himmelblau, [x_next])
        y_next = y_next[0]

        X_history.append(x_next.tolist())
        y_history.append(y_next)
        best_history.append(min(best_history[-1], y_next))

        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"  Iter {iteration + 1:3d}/{num_iter}: "
                  f"best={best_history[-1]:.4f}, "
                  f"eval=({x_next[0]:.3f}, {x_next[1]:.3f}), "
                  f"time={elapsed:.3f}s")

    return {
        "X_history": np.array(X_history),
        "y_history": np.array(y_history),
        "best_history": best_history,
        "times": times,
        "gp": gp,
    }


def bayesian_optimizer_batch(num_init: int, num_iter: int, num_workers: int,
                             bounds: list, seed: int = 42) -> dict:
    """
    Batch Bayesian Optimization: evaluate multiple candidates per iteration.

    In each iteration:
      1. Fit GP
      2. Select top-K candidates from EI (with diversity)
      3. Evaluate all K in parallel via pool.map()
      4. Update GP

    This demonstrates parallelizing the likelihood computation for
    multiple candidate θ values simultaneously.
    """
    rng = default_rng(seed)
    dim = 2
    lo, hi = bounds[0], bounds[1]

    X_history = []
    y_history = []
    best_history = []
    times = []

    gp = GaussianProcess(length_scales=np.array([2.0, 2.0]))

    # --- Initial design (parallel) ---
    X_init = rng.uniform(lo, hi, (num_init, dim))
    with multiprocessing.Pool(processes=num_workers) as pool:
        y_init = pool.map(himmelblau, X_init)
    y_init = np.array(y_init)

    X_history.extend(X_init.tolist())
    y_history.extend(y_init.tolist())
    best_history.append(np.min(y_init))
    print(f"  Initial design: {num_init} points evaluated")

    # --- Iterative batch BO ---
    batch_size = min(num_workers, 4)  # Limit batch size for grid search

    for iteration in range(num_iter):
        t0 = time.perf_counter()

        # Fit GP
        X_arr = np.array(X_history)
        y_arr = np.array(y_history)
        gp.fit(X_arr, y_arr)
        f_best = np.min(y_arr)

        # Grid search for EI candidates
        grid_size = 150
        x1_grid = np.linspace(lo, hi, int(np.sqrt(grid_size)))
        x2_grid = np.linspace(lo, hi, int(np.sqrt(grid_size)))
        X_candidates = np.array([[x1, x2] for x1 in x1_grid for x2 in x2_grid])

        mu, sigma = gp.predict(X_candidates)
        ei = expected_improvement(mu, sigma, f_best)

        # Select top-K candidates with diversity (simple top-K)
        top_k_indices = np.argsort(ei)[-batch_size:]
        candidates = X_candidates[top_k_indices]

        # Parallel evaluation of all candidates
        with multiprocessing.Pool(processes=num_workers) as pool:
            y_new = pool.map(himmelblau, candidates)
        y_new = np.array(y_new)

        for x, y in zip(candidates, y_new):
            X_history.append(x.tolist())
            y_history.append(y)
        best_history.append(min(best_history[-1], np.min(y_new)))

        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"  Iter {iteration + 1:3d}/{num_iter}: "
                  f"best={best_history[-1]:.4f}, "
                  f"evals={batch_size}, "
                  f"time={elapsed:.3f}s")

    return {
        "X_history": np.array(X_history),
        "y_history": np.array(y_history),
        "best_history": best_history,
        "times": times,
        "gp": gp,
    }


# ===================================================================
# Visualization
# ===================================================================

def plot_bo_trajectory(X_history: np.ndarray, y_history: np.ndarray,
                       best_history: list, bounds: list,
                       output_dir: str = ".") -> str:
    """Contour plot of Himmelblau's function with BO trajectory."""
    x_range = np.linspace(bounds[0], bounds[1], 300)
    y_range = np.linspace(bounds[0], bounds[1], 300)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([himmelblau(np.array([x, y]))
                  for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    # Log scale for better visualization
    Z_safe = np.where(Z > 0, Z, 1e-10)

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, np.log1p(Z_safe), levels=50, cmap="viridis")
    plt.colorbar(contour, label="log(1 + f(x, y))", ax=ax)

    # Plot BO trajectory
    n = len(X_history)
    t = np.linspace(0, 1, n)
    colors = plt.cm.plasma(t)
    ax.scatter(X_history[:, 0], X_history[:, 1], c=colors, s=40,
               edgecolors="white", linewidths=0.5, zorder=3, label="BO evaluations")

    # Mark global minima
    minima = [[3.0, 2.0], [-2.805, 3.131], [-3.779, -3.283], [3.584, -1.848]]
    for mx, my in minima:
        ax.scatter([mx], [my], c="red", s=80, marker="*", zorder=5,
                   label="Global minima" if mx == minima[0][0] else "")

    # Mark first and last
    ax.scatter([X_history[0, 0]], [X_history[0, 1]], c="green", s=100,
               marker="o", edgecolors="black", zorder=4, label="First evaluation")
    ax.scatter([X_history[-1, 0]], [X_history[-1, 1]], c="magenta", s=100,
               marker="*", edgecolors="black", zorder=4, label="Last evaluation")

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Bayesian Optimization: Search Trajectory", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    path = f"{output_dir}/bo_trajectory.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bo_convergence(best_history: list, output_dir: str = ".") -> str:
    """Plot best objective value vs. iteration."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(best_history, "bo-", linewidth=2, markersize=6)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Objective Value", fontsize=12)
    ax.set_title("Bayesian Optimization: Convergence", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    path = f"{output_dir}/bo_convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_acquisition_function(gp: GaussianProcess, X_history: np.ndarray,
                              y_history: np.ndarray, bounds: list,
                              output_dir: str = ".") -> str:
    """
    Plot GP posterior and acquisition function on a 1D slice.

    Shows the posterior mean ± 2σ and the EI acquisition function,
    illustrating the exploration-exploitation trade-off.
    """
    # Fix y=0, vary x
    x_slice = np.linspace(bounds[0], bounds[1], 200)
    X_slice = np.column_stack([x_slice, np.zeros_like(x_slice)])

    mu, sigma = gp.predict(X_slice)
    ei = expected_improvement(mu, sigma, np.min(y_history))

    fig, ax = plt.subplots(figsize=(12, 5))

    # GP posterior
    ax.plot(x_slice, mu, "b-", linewidth=2, label="GP posterior mean μ*(x)")
    ax.fill_between(x_slice, mu - 2 * sigma, mu + 2 * sigma,
                    alpha=0.2, color="blue", label="±2σ uncertainty")

    # Acquisition function
    ax2 = ax.twinx()
    ax2.plot(x_slice, ei, "r--", linewidth=2, label="EI acquisition function")

    # Plot training data
    ax.scatter(X_history[:, 0], X_history[:, 1], c="black", s=30, zorder=3)

    ax.set_xlabel("x (y=0 slice)", fontsize=12)
    ax.set_ylabel("GP Posterior", fontsize=12)
    ax2.set_ylabel("EI(x)", fontsize=12)
    ax.set_title("GP Posterior & Expected Improvement", fontsize=14)

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)

    path = f"{output_dir}/bo_acquisition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("Exercise 2-2: Bayesian Optimization with multiprocessing")
    print("=" * 60)
    print()

    # --- Parameters ---
    num_init = 10
    num_iter = 40
    bounds = [-5.0, 5.0]
    worker_counts = [1, 2, 4]

    print(f"Optimization problem: Himmelblau's function (2D)")
    print(f"  f(x,y) = (x²+y-11)² + (x+y²-7)²")
    print(f"  Four global minima at f=0")
    print()
    print(f"BO parameters: init={num_init}, iterations={num_iter}")
    print()

    # --- Bayes' theorem explanation ---
    print("[1] Bayes' theorem in Bayesian Optimization")
    print()
    print("    GP Posterior = Bayes' theorem for functions:")
    print()
    print("      p(f | D) = p(D | f) · p(f) / p(D)")
    print("      posterior    likelihood   prior   evidence")
    print()
    print("    where:")
    print("      p(f)        = GP prior over functions")
    print("      p(D | f)    = likelihood of observations")
    print("      p(f | D)    = GP posterior (Bayes' update)")
    print()
    print("    Closed-form GP update:")
    print("      μ*(x) = k(x,X) · (K + σ²I)⁻¹ y    ← posterior mean")
    print("      σ*²(x) = k(x,x) - k(x,X)·K⁻¹·k(x,X) ← posterior variance")
    print()
    print("    Expected Improvement = expectation under posterior:")
    print("      EI(x) = E[max(0, f* - f(x))]  ← Monte Carlo approx")
    print("            = analytical formula (exact for Gaussian)")
    print()
    print("    Parallelization: pool.map(himmelblau, candidates)")
    print("    evaluates multiple θ candidates simultaneously — the")
    print("    likelihood computation p(D|θ) for many θ values.")
    print()

    # --- Main BO run ---
    print("[2] Running Bayesian Optimization (workers=4, batch) ...")
    start = time.perf_counter()
    result = bayesian_optimizer_batch(
        num_init=num_init, num_iter=num_iter, num_workers=4,
        bounds=bounds, seed=42
    )
    elapsed = time.perf_counter() - start
    print(f"  Final best = {result['best_history'][-1]:.6f}")
    print(f"  Total time: {elapsed:.3f}s")
    print()

    # --- Visualization ---
    print("[3] Generating trajectory plot ...")
    traj_path = plot_bo_trajectory(
        result["X_history"], result["y_history"],
        result["best_history"], bounds
    )
    print(f"  Saved: {traj_path}")

    print("[4] Generating convergence plot ...")
    conv_path = plot_bo_convergence(result["best_history"])
    print(f"  Saved: {conv_path}")

    print("[5] Generating acquisition function plot ...")
    acq_path = plot_acquisition_function(
        result["gp"], result["X_history"], result["y_history"], bounds
    )
    print(f"  Saved: {acq_path}")
    print()

    # --- Parallel speedup ---
    print("[6] Parallel speedup comparison")
    speedup_results = []
    for w in worker_counts:
        start = time.perf_counter()
        res = bayesian_optimizer_batch(
            num_init=num_init, num_iter=num_iter, num_workers=w,
            bounds=bounds, seed=42
        )
        elapsed = time.perf_counter() - start
        speedup_results.append({
            "workers": w, "time": elapsed,
            "best_fitness": res["best_history"][-1]
        })
        print(f"  workers={w}: time={elapsed:.3f}s, best={res['best_history'][-1]:.6f}")

    print()
    print("[7] Summary: Bayes' theorem parallelization")
    print("    Bayesian Optimization applies Bayes' theorem to function")
    print("    optimization:")
    print()
    print("      1. GP posterior = Bayes' theorem (prior + likelihood)")
    print("      2. EI acquisition = expectation under posterior")
    print("      3. Parallel evaluation = likelihood for multiple θ")
    print()
    print("    The black-box function evaluation is the likelihood p(D|θ).")
    print("    Each evaluation is independent → pool.map() parallelizes it.")
    print("    This is the same pattern as:")
    print("      - Particle filter: likelihood for each particle")
    print("      - GA: fitness (= likelihood) for each individual")
    print()
    print("Done.")


if __name__ == "__main__":
    main()