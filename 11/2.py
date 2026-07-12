import time
import multiprocessing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

# ===================================================================
# State-space model
# ===================================================================

def generate_true_state(T: int, x0: float, Q: float, rng: np.random.Generator) -> np.ndarray:
    """Generate the true state sequence from the state transition model."""
    x = np.zeros(T)
    x[0] = x0
    for t in range(1, T):
        x[t] = 0.5 * x[t - 1] + 10.0 * np.sin(0.1 * t) + rng.normal(0, np.sqrt(Q))
    return x


def generate_observations(T: int, true_state: np.ndarray, R: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Generate noisy observations from the observation model."""
    y = np.zeros(T)
    for t in range(T):
        y[t] = true_state[t] / (1.0 + 0.01 * true_state[t] ** 2) + rng.normal(0, np.sqrt(R))
    return y

# ===================================================================
# Particle filter core (Bayes' theorem in action)
# ===================================================================

def predict_particle(x_prev: float, t: int, Q: float, rng: np.random.Generator) -> float:
    """
    Sample a predicted particle from the state transition model.

    This computes p(x_t | x_{t-1}) — the prior predictive distribution.
    State transition: x_t = 0.5 * x_{t-1} + 10 * sin(0.1 * t) + w_t
    Each particle is sampled independently, making this step parallelizable.
    """
    return 0.5 * x_prev + 10.0 * np.sin(0.1 * t) + rng.normal(0, np.sqrt(Q))


def compute_log_likelihood(y_obs: float, x_pred: float, R: float) -> float:
    """
    Compute log p(y_t | x_t) — the observation likelihood.

    This is the LIKELIHOOD term in Bayes' theorem:
        p(x_t | y_{1:t}) ∝ p(y_t | x_t) · p(x_t | y_{1:t-1})

    For Gaussian observation noise:
        p(y_t | x_t) = N(y_t; h(x_t), R)
        log p(y_t | x_t) = -0.5 * log(2πR) - 0.5 * (y_t - h(x_t))^2 / R

    where h(x_t) = x_t / (1 + 0.01 * x_t^2) is the observation function.

    This function is called independently for each particle, making it
    ideal for parallelization via pool.map().
    """
    h_x = x_pred / (1.0 + 0.01 * x_pred ** 2)
    return -0.5 * np.log(2 * np.pi * R) - 0.5 * (y_obs - h_x) ** 2 / R


def log_sum_exp(log_weights: np.ndarray) -> float:
    """Numerically stable log-sum-exp to avoid overflow/underflow."""
    max_val = np.max(log_weights)
    return max_val + np.log(np.sum(np.exp(log_weights - max_val)))


def systematic_resample(log_weights: np.ndarray) -> np.ndarray:
    """
    Systematic resampling.

    When particle weights become degenerate (most weight on few particles),
    resample to maintain diversity. This is the only sequential step in the PF.
    """
    N = len(log_weights)
    cumulative = np.exp(log_weights - log_sum_exp(log_weights))
    cumulative = np.cumsum(cumulative)
    cumulative[-1] = 1.0  # Fix floating point

    u = np.arange(N) / N + np.random.uniform(0, 1 / N)
    indices = np.searchsorted(cumulative, u)
    return indices

def particle_filter_step_serial(particles: np.ndarray, weights_log: np.ndarray,
                                y_obs: float, t: int, Q: float, R: float,
                                prior_std: float, rng: np.random.Generator) -> tuple:
    """
    Single time-step of the SIR particle filter (serial version).

    Steps:
      1. Predict: sample new particles from the state transition model
      2. Update: compute log-likelihoods (Bayes' theorem likelihood term)
      3. Normalize weights
      4. Resample if needed

    This is the serial reference. The parallel version dispatches steps 1-2
    to a multiprocessing pool.
    """
    N = len(particles)

    # Step 1: Predict — sample from transition model
    new_particles = np.array([
        predict_particle(particles[i], t, Q, rng) for i in range(N)
    ])

    # Step 2: Update — compute log-likelihoods (Bayes' theorem)
    log_likelihoods = np.array([
        compute_log_likelihood(y_obs, new_particles[i], R) for i in range(N)
    ])

    # Bayes' theorem: log w_t ∝ log w_{t-1} + log p(y_t | x_t)
    new_log_weights = weights_log + log_likelihoods

    # Step 3: Normalize
    log_sum = log_sum_exp(new_log_weights)
    new_log_weights = new_log_weights - log_sum

    # Step 4: Resample
    if _resampling_needed(new_log_weights):
        indices = systematic_resample(new_log_weights)
        new_particles = new_particles[indices]
        new_log_weights = np.full(N, -np.log(N))

    return new_particles, new_log_weights

def _resampling_needed(log_weights: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Check if effective sample size (ESS) is too low.

    ESS = exp(sum(log w) - log_sum_exp(log w * 2))
         = sum(w_i^2) / (sum(w_i))^2  (in probability space)

    Resample when ESS < threshold * N.
    """
    N = len(log_weights)
    # Correct ESS formula: exp(sum(log w) - log_sum_exp(2 * log w))
    ess = np.exp(log_sum_exp(log_weights) - log_sum_exp(log_weights * 2))
    return ess < threshold * N

def particle_filter_parallel(y: np.ndarray, N_particles: int, num_workers: int,
                             Q: float, R: float, x0_prior: float,
                             prior_std: float, seed: int = 42,
                             true_state: np.ndarray = None) -> dict:
    """
    Full SIR particle filter with parallel particle updates.

    Bayes' theorem is applied at each time step:
        posterior ∝ likelihood × prior

    The likelihood p(y_t | x_t) is computed in parallel for all particles
    using multiprocessing.Pool.map().

    Args:
        y:            Observations array of shape (T,)
        N_particles:  Number of particles
        num_workers:  Number of parallel workers
        Q:            Process noise variance
        R:            Observation noise variance
        x0_prior:     Initial state prior mean
        prior_std:    Initial state prior standard deviation
        seed:         Random seed for reproducibility

    Returns:
        Dictionary with trajectories, weights, and RMSE.
    """
    rng = default_rng(seed)
    T = len(y)
    N = N_particles

    # Initialize particles from prior: p(x_0) = N(x0_prior, prior_std^2)
    particles = rng.normal(x0_prior, prior_std, N)
    log_weights = np.full(N, -np.log(N))  # Uniform initial weights

    # Storage
    trajectories = np.zeros((T, N))
    all_log_weights = np.zeros((T, N))

    # Time stepping
    for t in range(T):
        trajectories[t] = particles
        all_log_weights[t] = log_weights

        if t == 0:
            continue  # No update for t=0 (first observation initializes weights)

        y_obs = y[t]

        # --- Parallel predict + update (Bayes' theorem) ---
        # Each worker handles a subset of particles
        # We split particles into chunks for the pool

        chunk_size = (N + num_workers - 1) // num_workers
        chunks = []
        for i in range(num_workers):
            start = i * chunk_size
            end = min(start + chunk_size, N)
            if start < N:
                chunk_indices = list(range(start, end))
                chunk_particles = particles[chunk_indices].copy()
                chunks.append((chunk_particles, y_obs, t, Q, R))

        # Parallel predict + update — the core of Bayes' theorem
        # Each worker processes a chunk: predicts new state and computes log-likelihood
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.starmap(_predict_chunk, chunks)

        # Collect results: results = [(predicted_chunk, log_likes_chunk), ...]
        new_particles = np.zeros(N)
        new_log_weights = np.zeros(N)
        offset = 0
        for chunk_particles, chunk_log_likes in results:
            n_chunk = len(chunk_particles)
            new_particles[offset:offset + n_chunk] = chunk_particles
            # Bayes' theorem: log posterior = log prior + log likelihood
            new_log_weights[offset:offset + n_chunk] = (
                log_weights[offset:offset + n_chunk] + chunk_log_likes
            )
            offset += n_chunk

        # Normalize
        log_sum = log_sum_exp(new_log_weights)
        new_log_weights = new_log_weights - log_sum

        # Resample
        if _resampling_needed(new_log_weights):
            indices = systematic_resample(new_log_weights)
            new_particles = new_particles[indices]
            new_log_weights = np.full(N, -np.log(N))

        particles = new_particles
        log_weights = new_log_weights

    # Compute estimates and RMSE
    estimates = np.mean(trajectories, axis=1)
    if true_state is not None:
        rmse = np.sqrt(np.mean((estimates - true_state) ** 2))
    else:
        rmse = float("nan")  # True state not provided

    return {
        "trajectories": trajectories,
        "log_weights": all_log_weights,
        "estimates": estimates,
        "rmse": rmse,
    }

def _predict_chunk(chunk_particles: np.ndarray, y_obs: float, t: int, Q: float,
                   R: float) -> tuple:
    """
    Worker function: predict + compute log-likelihoods for a chunk of particles.

    Each worker gets its own RNG so results are independent.
    State transition: x_t = 0.5 * x_{t-1} + 10 * sin(0.1 * t) + w_t
    Returns (predicted_particles, log_likelihoods).
    """
    rng = default_rng()
    predicted = 0.5 * chunk_particles + 10.0 * np.sin(0.1 * t) + rng.normal(0, np.sqrt(Q), len(chunk_particles))
    log_likes = np.array([compute_log_likelihood(y_obs, p, R) for p in predicted])
    return predicted, log_likes

# ===================================================================
# Visualization
# ===================================================================

def plot_trajectory(true_state: np.ndarray, observations: np.ndarray,
                    result: dict, output_dir: str = ".") -> str:
    """Plot particle filter trajectory with observation overlay."""
    T = len(true_state)
    trajectories = result["trajectories"]
    estimates = result["estimates"]

    # Compute particle std for error band
    std = np.std(trajectories, axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(true_state, "k-", linewidth=2, label="True state")
    ax.scatter(range(T), observations, c="red", s=15, alpha=0.5, label="Observations")
    ax.plot(estimates, "b--", linewidth=1.5, label="PF estimate (mean)")
    ax.fill_between(range(T), estimates - 2 * std, estimates + 2 * std,
                    alpha=0.2, color="blue", label="±2σ particle cloud")

    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("State", fontsize=12)
    ax.set_title("Particle Filter: State Estimation", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = f"{output_dir}/particle_filter_trajectory.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_parameter_sweep(results_grid: dict, output_dir: str = ".") -> str:
    """
    2D heatmap: N_particles × num_workers → RMSE and execution time.

    This shows the trade-off surface between accuracy and speed.
    """
    particle_counts = sorted(results_grid.keys())
    worker_counts = sorted(next(iter(results_grid.values())).keys())

    rmse_grid = np.zeros((len(particle_counts), len(worker_counts)))
    time_grid = np.zeros((len(particle_counts), len(worker_counts)))

    for i, n in enumerate(particle_counts):
        for j, w in enumerate(worker_counts):
            rmse_grid[i, j] = results_grid[n][w]["rmse"]
            time_grid[i, j] = results_grid[n][w]["time"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RMSE heatmap
    im1 = axes[0].imshow(rmse_grid, aspect="auto", origin="lower",
                         cmap="hot_r", extent=[
                             worker_counts[0] - 0.5, worker_counts[-1] + 0.5,
                             particle_counts[0] - 0.5, particle_counts[-1] + 0.5
                         ])
    axes[0].set_xlabel("Number of Workers", fontsize=12)
    axes[0].set_ylabel("Number of Particles", fontsize=12)
    axes[0].set_title("RMSE: Accuracy", fontsize=14)
    axes[0].invert_yaxis()
    plt.colorbar(im1, ax=axes[0], label="RMSE")

    # Annotate
    for i in range(len(particle_counts)):
        for j in range(len(worker_counts)):
            axes[0].text(j, i, f"{rmse_grid[i, j]:.3f}", ha="center", va="center",
                         fontsize=8, color="white" if rmse_grid[i, j] > rmse_grid.max() * 0.5 else "black")

    # Time heatmap
    im2 = axes[1].imshow(time_grid, aspect="auto", origin="lower",
                         cmap="viridis", extent=[
                             worker_counts[0] - 0.5, worker_counts[-1] + 0.5,
                             particle_counts[0] - 0.5, particle_counts[-1] + 0.5
                         ])
    axes[1].set_xlabel("Number of Workers", fontsize=12)
    axes[1].set_title("Execution Time (s)", fontsize=14)
    plt.colorbar(im2, ax=axes[1], label="Time (s)")

    for i in range(len(particle_counts)):
        for j in range(len(worker_counts)):
            axes[1].text(j, i, f"{time_grid[i, j]:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if time_grid[i, j] > time_grid.max() * 0.5 else "black")

    path = f"{output_dir}/particle_filter_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("Exercise 11-2: Particle Filter with multiprocessing")
    print("=" * 60)
    print()

    # --- Parameters ---
    T = 50            # Time steps
    x0_true = 0.0     # True initial state
    Q = 1.0           # Process noise
    R = 4.0           # Observation noise
    x0_prior = 0.0    # Prior mean
    prior_std = 5.0   # Prior std

    print(f"State-space model:")
    print(f"  x_t = 0.5*x_{{t-1}} + 10*sin(0.1*t) + w_t,  w_t ~ N(0, {Q})")
    print(f"  y_t = x_t / (1 + 0.01*x_t^2) + v_t,        v_t ~ N(0, {R})")
    print()

    # --- Generate data ---
    rng = default_rng(42)
    true_state = generate_true_state(T, x0_true, Q, rng)
    observations = generate_observations(T, true_state, R, rng)
    print(f"Generated {T} time steps of observations")
    print()

    # --- Bayes' theorem explanation ---
    print("[1] Bayes' theorem in the Particle Filter")
    print("    p(x_t | y_{1:t}) ∝ p(y_t | x_t) · p(x_t | y_{1:t-1})")
    print()
    print("    where:")
    print("      p(y_t | x_t)        = N(y_t; h(x_t), R)   ← LIKELIHOOD (parallel)")
    print("      p(x_t | y_{1:t-1})  = predicted particles   ← PRIOR")
    print("      p(x_t | y_{1:t})    = weighted particles    ← POSTERIOR")
    print()
    print("    Each particle's weight is updated by multiplying by the")
    print("    likelihood p(y_t | x_t^{(i)}) — this is the core of Bayes'")
    print("    theorem. All likelihood computations are independent → parallel.")
    print()

    # --- Single run visualization ---
    print("[2] Running particle filter (N=200, workers=4) ...")
    start = time.perf_counter()
    result = particle_filter_parallel(
        y=observations, N_particles=200, num_workers=4,
        Q=Q, R=R, x0_prior=x0_prior, prior_std=prior_std,
        true_state=true_state
    )
    elapsed = time.perf_counter() - start
    print(f"  RMSE = {result['rmse']:.4f}  (time: {elapsed:.3f}s)")
    print()

    # --- Visualization ---
    print("[3] Generating trajectory plot ...")
    traj_path = plot_trajectory(true_state, observations, result)
    print(f"  Saved: {traj_path}")
    print()

    # --- Parameter sweep ---
    print("[4] Parameter sweep: N_particles × num_workers")
    particle_counts = [10, 50, 200, 800]
    worker_counts = [1, 2, 4, 8]

    results_grid = {}
    for n in particle_counts:
        results_grid[n] = {}
        for w in worker_counts:
            start = time.perf_counter()
            res = particle_filter_parallel(
                y=observations, N_particles=n, num_workers=w,
                Q=Q, R=R, x0_prior=x0_prior, prior_std=prior_std,
                true_state=true_state
            )
            elapsed = time.perf_counter() - start
            results_grid[n][w] = {"rmse": res["rmse"], "time": elapsed}
            print(f"  N={n:4d}  workers={w}: RMSE={res['rmse']:.4f}  time={elapsed:.3f}s")

    print()
    print("[5] Generating parameter sweep heatmap ...")
    sweep_path = plot_parameter_sweep(results_grid)
    print(f"  Saved: {sweep_path}")
    print()

    # --- Summary ---
    print("[6] Summary: Bayes' theorem parallelization")
    print("    The likelihood computation p(y_t | x_t^{(i)}) for each particle")
    print("    is independent of other particles. This is exactly the kind of")
    print("    independent computation that Bayes' theorem requires when")
    print("    evaluating the likelihood for multiple data points.")
    print()
    print("    In Exercise 12-2 (Bayesian Optimization), the same pattern")
    print("    parallelizes the black-box likelihood evaluations.")
    print()
    print("Done.")

if __name__ == "__main__":
    main()