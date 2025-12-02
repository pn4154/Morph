"""
Evaluation script for comparing PPO model against baselines.
Compares adaptive partitioning (PPO) vs static partitions vs unpartitioned.
"""

import sys
import os
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO
from env import PartitionEnv
from workloads import (
    GaussianWorkload,
    SlidingGaussianWorkload,
    BimodalWorkload,
    UniformWorkload,
)
import db_utils
import init_db
from workloads.BimodalRange import BimodalRangeWorkload
from workloads.GaussianRange import GaussianRangeWorkload
from workloads.SlidingGaussianRange import SlidingGaussianRangeWorkload
from workloads.UniformRange import UniformRangeWorkload

# Constants
LATENCY_SMOOTHING_WINDOW = 50  # Smooth latencies over this many queries
NUM_RUNS = 5  # Number of runs per workload type
NUM_ITERATIONS = 20  # Number of iterations per run
QUERIES_PER_ITERATION = 50  # Queries per iteration
NUM_ORDERS = 10000  # Maximum OrderID in database


def initialize_database():
    """Reinitialize the database to clean state."""
    print("  Reinitializing database...")
    init_db.drop_and_create_database()

    conn = db_utils.get_connection()
    init_db.create_tables(conn)
    init_db.populate_tables(conn)
    conn.close()
    print("   Database reinitialized")


def evaluate_ppo_model(model_path, workload, run_num):
    """Evaluate PPO model with adaptive partitioning.

    Args:
        model_path: Path to saved PPO model
        workload: Workload instance to use
        run_num: Run number for logging

    Returns:
        tuple: (all_latencies, total_repartition_time)
    """
    print(f"    Run {run_num + 1}: PPO Model")

    # Load model
    model = PPO.load(model_path)

    # Create environment with specific workload
    env = PartitionEnv(
        workload=workload, num_orders=NUM_ORDERS, queries_per_step=QUERIES_PER_ITERATION
    )

    all_latencies = []
    total_repartition_time = 0.0

    # Reset environment
    obs, info = env.reset()

    # Run iterations
    for iteration in tqdm(
        range(NUM_ITERATIONS), desc="      PPO iterations", leave=False
    ):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)

        # Time the step (includes repartitioning + queries)
        step_start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_end = time.time()

        # Extract latencies from environment
        iteration_latencies = env.last_latencies.copy()
        all_latencies.extend(iteration_latencies)

        # Estimate repartition time (step time - query time)
        query_time = sum(iteration_latencies) / 1000.0  # Convert ms to seconds
        repartition_time = max(0, step_end - step_start - query_time)
        total_repartition_time += repartition_time

    env.close()

    print(f"      Total repartition time: {total_repartition_time:.3f}s")
    print(f"      Avg query latency: {np.mean(all_latencies):.3f}ms")

    return all_latencies, total_repartition_time


def evaluate_static_partitions(workload, run_num):
    """Evaluate static equal partitions (1/3 boundaries).

    Args:
        workload: Workload instance to use
        run_num: Run number for logging

    Returns:
        list: all_latencies
    """
    print(f"    Run {run_num + 1}: Static Partitions")

    # Get connection
    conn = db_utils.get_connection()

    # Set static partitions at 1/3 boundaries
    db_utils.repartition(conn, 0.333, 0.666)
    print("      Partitioned at 1/3 boundaries")

    all_latencies = []
    total_queries = NUM_ITERATIONS * QUERIES_PER_ITERATION

    # Run all queries without repartitioning
    for query_idx in tqdm(
        range(total_queries), desc="      Static queries", leave=False
    ):
        query = workload.sample(NUM_ORDERS)
        _, latency_ms = db_utils.execute_query(conn, query, None)
        all_latencies.append(latency_ms)

    conn.close()

    print(f"      Avg query latency: {np.mean(all_latencies):.3f}ms")

    return all_latencies


def evaluate_unpartitioned(workload, run_num):
    """Evaluate unpartitioned database (single large partition).

    Args:
        workload: Workload instance to use
        run_num: Run number for logging

    Returns:
        list: all_latencies
    """
    print(f"    Run {run_num + 1}: Unpartitioned")

    # Get connection
    conn = db_utils.get_connection()

    # Create "unpartitioned" setup - one big partition covering everything
    # We use boundaries at 0.001 and 0.999 so essentially all data is in middle partition
    db_utils.repartition(conn, 0.001, 0.999)
    print("      Using single large partition (unpartitioned)")

    all_latencies = []
    total_queries = NUM_ITERATIONS * QUERIES_PER_ITERATION

    # Run all queries
    for query_idx in tqdm(
        range(total_queries), desc="      Unpartitioned queries", leave=False
    ):
        query = workload.sample(NUM_ORDERS)
        _, latency_ms = db_utils.execute_query(conn, query, None)
        all_latencies.append(latency_ms)

    conn.close()

    print(f"      Avg query latency: {np.mean(all_latencies):.3f}ms")

    return all_latencies


def smooth_latencies(latencies, window_size):
    """Apply moving average smoothing to latencies.

    Args:
        latencies: List of latency values
        window_size: Size of smoothing window

    Returns:
        np.array: Smoothed latencies
    """
    if len(latencies) < window_size:
        return np.array(latencies)

    smoothed = np.convolve(latencies, np.ones(window_size) / window_size, mode="valid")
    return smoothed


def create_comparison_graphs(results, output_dir):
    """Create comparison graphs for each workload type.

    Args:
        results: Dictionary with structure:
                {workload_name: {
                    'ppo': [[latencies_run1], [latencies_run2], ...],
                    'static': [[latencies_run1], ...],
                    'unpartitioned': [[latencies_run1], ...]
                }}
        output_dir: Directory to save graphs
    """
    print("\n=== Creating Comparison Graphs ===")

    workload_names = list(results.keys())
    colors = {"ppo": "blue", "static": "green", "unpartitioned": "red"}
    labels = {
        "ppo": "PPO (Adaptive)",
        "static": "Static (1/3)",
        "unpartitioned": "Unpartitioned",
    }

    for workload_name in workload_names:
        print(f"  Creating graph for {workload_name}...")

        plt.figure(figsize=(12, 6))

        # Plot each model type
        for model_type in ["ppo", "static", "unpartitioned"]:
            runs_data = results[workload_name][model_type]

            # Plot each run
            for run_idx, latencies in enumerate(runs_data):
                # Smooth latencies
                smoothed = smooth_latencies(latencies, LATENCY_SMOOTHING_WINDOW)

                # X-axis starts after smoothing window
                x_values = np.arange(LATENCY_SMOOTHING_WINDOW - 1, len(latencies))

                # Plot with alpha for transparency
                label = labels[model_type] if run_idx == 0 else None
                plt.plot(
                    x_values,
                    smoothed,
                    color=colors[model_type],
                    alpha=0.6,
                    linewidth=1,
                    label=label,
                )

        plt.xlabel("Query Index")
        plt.ylabel(f"Latency (ms) - Smoothed over {LATENCY_SMOOTHING_WINDOW} queries")
        plt.title(f"{workload_name} Workload: PPO vs Static vs Unpartitioned")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save graph
        filename = os.path.join(output_dir, f"{workload_name.lower()}_comparison.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"     Saved to {filename}")


def run_evaluation(model_path):
    """Run full evaluation comparing PPO vs static vs unpartitioned.

    Args:
        model_path: Path to trained PPO model (.zip file)
    """
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Runs per workload: {NUM_RUNS}")
    print(f"Iterations per run: {NUM_ITERATIONS}")
    print(f"Queries per iteration: {QUERIES_PER_ITERATION}")
    print(f"Total queries per run: {NUM_ITERATIONS * QUERIES_PER_ITERATION}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("graphs", f"evaluation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Define workload types
    workload_types = {
        "Gaussian": GaussianWorkload,
        "SlidingGaussian": SlidingGaussianWorkload,
        "Bimodal": BimodalWorkload,
        "Uniform": UniformWorkload,
        "GaussianRange": GaussianRangeWorkload,
        "SlidingGaussianRange": SlidingGaussianRangeWorkload,
        "BimodalRange": BimodalRangeWorkload,
        "UniformRange": UniformRangeWorkload,
    }

    # Store results for graphing
    results = {
        name: {"ppo": [], "static": [], "unpartitioned": []}
        for name in workload_types.keys()
    }

    # Run evaluation
    print(
        f"\nStarting evaluation with {NUM_RUNS} runs × {len(workload_types)} workloads × 3 approaches..."
    )
    total_evals = NUM_RUNS * len(workload_types) * 3

    with tqdm(total=total_evals, desc="Overall Progress") as pbar:
        for run_num in range(NUM_RUNS):
            for workload_name, WorkloadClass in workload_types.items():
                # Create workload instance with random parameters
                workload = WorkloadClass()

                # 1. Evaluate PPO model
                pbar.set_description(
                    f"Run {run_num+1}/{NUM_RUNS} - {workload_name} - PPO"
                )
                initialize_database()
                ppo_latencies, repartition_time = evaluate_ppo_model(
                    model_path, workload, run_num
                )
                results[workload_name]["ppo"].append(ppo_latencies)
                pbar.update(1)

                # 2. Evaluate static partitions
                pbar.set_description(
                    f"Run {run_num+1}/{NUM_RUNS} - {workload_name} - Static"
                )
                initialize_database()
                static_latencies = evaluate_static_partitions(workload, run_num)
                results[workload_name]["static"].append(static_latencies)
                pbar.update(1)

                # 3. Evaluate unpartitioned
                pbar.set_description(
                    f"Run {run_num+1}/{NUM_RUNS} - {workload_name} - Unpartitioned"
                )
                initialize_database()
                unpartitioned_latencies = evaluate_unpartitioned(workload, run_num)
                results[workload_name]["unpartitioned"].append(unpartitioned_latencies)
                pbar.update(1)

    # Create comparison graphs
    create_comparison_graphs(results, output_dir)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for workload_name in workload_types.keys():
        print(f"\n{workload_name} Workload:")

        for model_type in ["ppo", "static", "unpartitioned"]:
            all_runs = results[workload_name][model_type]
            # Flatten all runs
            all_latencies = [lat for run in all_runs for lat in run]
            avg = np.mean(all_latencies)
            std = np.std(all_latencies)
            print(f"  {model_type.upper():15s}: {avg:.3f} +/- {std:.3f} ms")

    print(f"\n{'=' * 60}")
    print(f"Evaluation complete! Graphs saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <model_path>")
        print("Example: python evaluate.py checkpoints/model_20231201_143022.zip")
        sys.exit(1)

    model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    run_evaluation(model_path)
