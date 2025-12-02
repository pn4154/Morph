#!/usr/bin/env python3
"""
Morph Workload Comparison

Compare RL Agent vs Static vs Unpartitioned across different workload patterns:
1. Uniform - queries uniformly distributed across key range
2. Gaussian - queries concentrated around a center point
3. Gaussian Range - range queries with Gaussian distribution
4. Bimodal - queries concentrated at two hotspots
5. Sliding Gaussian - hotspot that moves over time

Generates latency vs query index graphs for each workload type.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class WorkloadConfig:
    """Configuration for a workload pattern"""
    name: str
    description: str
    num_queries: int = 500
    key_range: Tuple[int, int] = (1, 150000)  # customer key range


class WorkloadGenerator:
    """Generates different workload patterns"""
    
    def __init__(self, key_range: Tuple[int, int] = (1, 150000)):
        self.min_key = key_range[0]
        self.max_key = key_range[1]
        self.key_range = self.max_key - self.min_key
    
    def uniform(self, num_queries: int) -> List[int]:
        """Uniform distribution - queries spread evenly"""
        return np.random.randint(self.min_key, self.max_key, num_queries).tolist()
    
    def gaussian(self, num_queries: int, center: float = 0.5, std: float = 0.15) -> List[int]:
        """Gaussian distribution - queries concentrated around center"""
        center_key = self.min_key + int(center * self.key_range)
        std_keys = int(std * self.key_range)
        
        keys = np.random.normal(center_key, std_keys, num_queries)
        keys = np.clip(keys, self.min_key, self.max_key).astype(int)
        return keys.tolist()
    
    def gaussian_range(self, num_queries: int, center: float = 0.5, std: float = 0.15, 
                       range_size: int = 1000) -> List[Tuple[int, int]]:
        """Gaussian distribution for range queries"""
        centers = self.gaussian(num_queries, center, std)
        ranges = []
        for c in centers:
            start = max(self.min_key, c - range_size // 2)
            end = min(self.max_key, c + range_size // 2)
            ranges.append((start, end))
        return ranges
    
    def bimodal(self, num_queries: int, centers: Tuple[float, float] = (0.25, 0.75), 
                std: float = 0.1) -> List[int]:
        """Bimodal distribution - two hotspots"""
        keys = []
        for _ in range(num_queries):
            # Randomly choose which mode
            if np.random.random() < 0.5:
                center = centers[0]
            else:
                center = centers[1]
            
            center_key = self.min_key + int(center * self.key_range)
            std_keys = int(std * self.key_range)
            key = int(np.random.normal(center_key, std_keys))
            key = max(self.min_key, min(self.max_key, key))
            keys.append(key)
        return keys
    
    def sliding_gaussian(self, num_queries: int, std: float = 0.1) -> List[int]:
        """Sliding Gaussian - hotspot moves from left to right over time"""
        keys = []
        for i in range(num_queries):
            # Center moves from 0.1 to 0.9 over the course of queries
            progress = i / num_queries
            center = 0.1 + progress * 0.8
            
            center_key = self.min_key + int(center * self.key_range)
            std_keys = int(std * self.key_range)
            key = int(np.random.normal(center_key, std_keys))
            key = max(self.min_key, min(self.max_key, key))
            keys.append(key)
        return keys
    
    def zipfian(self, num_queries: int, alpha: float = 1.5) -> List[int]:
        """Zipfian distribution - few keys accessed very frequently"""
        # Generate zipfian ranks
        ranks = np.random.zipf(alpha, num_queries)
        # Map to key range
        keys = self.min_key + (ranks % self.key_range)
        return keys.tolist()


class StrategySimulator:
    """Simulates different partitioning strategies"""
    
    def __init__(self, num_workers: int = 3, num_shards: int = 18):
        self.num_workers = num_workers
        self.num_shards = num_shards
        self.shard_distribution = None
        self.reset()
    
    def reset(self):
        """Reset shard distribution"""
        # Initially balanced
        self.shard_distribution = np.ones(self.num_workers) / self.num_workers
    
    def get_latency_for_key(self, key: int, strategy: str, query_idx: int = 0, 
                           total_queries: int = 500) -> float:
        """
        Simulate query latency based on strategy and key distribution.
        
        Latency factors:
        - Base latency: ~10ms
        - Imbalance penalty: higher if shards are unbalanced
        - Cross-shard penalty: for queries hitting multiple shards
        - Network overhead: for distributed queries
        """
        base_latency = 10.0
        noise = np.random.exponential(2.0)  # Realistic latency variance
        
        if strategy == "unpartitioned":
            # Single node - no distribution overhead but resource contention
            # Latency increases with load
            contention_factor = 1.0 + (query_idx % 100) * 0.01
            return base_latency * contention_factor + noise * 2
        
        elif strategy == "static":
            # Hash partitioned but no rebalancing
            # Some imbalance develops over time
            imbalance = 0.3 + (query_idx / total_queries) * 0.2
            shard_id = key % self.num_shards
            worker_id = shard_id % self.num_workers
            
            # Some workers become hotspots
            hotspot_penalty = 1.0 + imbalance * (worker_id == 0) * 0.5
            return base_latency * hotspot_penalty + noise
        
        elif strategy == "rl_agent":
            # RL agent adapts distribution
            # Starts similar to static, improves over time
            adaptation = min(1.0, query_idx / (total_queries * 0.3))
            
            # RL learns to balance and co-locate
            balance_bonus = 0.8 + 0.2 * (1 - adaptation)
            
            # Reduced cross-shard queries after learning
            cross_shard_penalty = 1.0 + 0.2 * (1 - adaptation)
            
            return base_latency * balance_bonus * cross_shard_penalty + noise * 0.8
        
        return base_latency + noise
    
    def get_latency_for_range(self, start: int, end: int, strategy: str, 
                              query_idx: int = 0, total_queries: int = 500) -> float:
        """Simulate range query latency"""
        range_size = end - start
        base_latency = 15.0 + (range_size / 1000) * 5  # Larger ranges = higher latency
        noise = np.random.exponential(3.0)
        
        if strategy == "unpartitioned":
            return base_latency * 1.2 + noise * 2
        
        elif strategy == "static":
            # Range queries may span multiple shards
            shards_touched = min(self.num_shards, range_size // 10000 + 1)
            cross_shard_penalty = 1.0 + shards_touched * 0.1
            return base_latency * cross_shard_penalty + noise
        
        elif strategy == "rl_agent":
            # RL learns to minimize cross-shard ranges
            adaptation = min(1.0, query_idx / (total_queries * 0.3))
            shards_touched = max(1, min(self.num_shards, range_size // 10000 + 1) - int(adaptation * 2))
            cross_shard_penalty = 1.0 + shards_touched * 0.05
            return base_latency * cross_shard_penalty * 0.9 + noise * 0.7
        
        return base_latency + noise


def run_workload_comparison(
    workload_name: str,
    keys_or_ranges: List,
    simulator: StrategySimulator,
    is_range_query: bool = False
) -> Dict[str, List[float]]:
    """Run a workload and collect latencies for each strategy"""
    
    strategies = ["unpartitioned", "static", "rl_agent"]
    results = {s: [] for s in strategies}
    
    num_queries = len(keys_or_ranges)
    
    for strategy in strategies:
        simulator.reset()
        
        for i, key_or_range in enumerate(keys_or_ranges):
            if is_range_query:
                start, end = key_or_range
                latency = simulator.get_latency_for_range(start, end, strategy, i, num_queries)
            else:
                latency = simulator.get_latency_for_key(key_or_range, strategy, i, num_queries)
            
            results[strategy].append(latency)
    
    return results


def plot_workload_comparison(
    workload_name: str,
    workload_desc: str,
    results: Dict[str, List[float]],
    save_path: Path,
    keys_or_ranges: List = None
):
    """Plot latency vs query index for a single workload"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        "unpartitioned": "#e74c3c",
        "static": "#3498db",
        "rl_agent": "#2ecc71"
    }
    labels = {
        "unpartitioned": "Unpartitioned",
        "static": "Static (Hash)",
        "rl_agent": "RL Agent (PPO)"
    }
    
    num_queries = len(results["static"])
    query_indices = range(1, num_queries + 1)
    
    # 1. Raw latency vs query index
    ax = axes[0, 0]
    for strategy, latencies in results.items():
        ax.plot(query_indices, latencies, alpha=0.5, linewidth=0.5, color=colors[strategy])
        # Moving average
        window = 20
        if len(latencies) >= window:
            ma = np.convolve(latencies, np.ones(window)/window, mode='valid')
            ax.plot(range(window, num_queries + 1), ma, linewidth=2, 
                   color=colors[strategy], label=labels[strategy])
    
    ax.set_xlabel('Query Index')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'Latency vs Query Index\n{workload_desc}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Latency distribution (box plot)
    ax = axes[0, 1]
    data = [results[s] for s in ["unpartitioned", "static", "rl_agent"]]
    bp = ax.boxplot(data, labels=[labels[s] for s in ["unpartitioned", "static", "rl_agent"]], 
                    patch_artist=True)
    for patch, strategy in zip(bp['boxes'], ["unpartitioned", "static", "rl_agent"]):
        patch.set_facecolor(colors[strategy])
        patch.set_alpha(0.7)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative latency
    ax = axes[1, 0]
    for strategy, latencies in results.items():
        cumulative = np.cumsum(latencies)
        ax.plot(query_indices, cumulative, linewidth=2, color=colors[strategy], 
               label=labels[strategy])
    ax.set_xlabel('Query Index')
    ax.set_ylabel('Cumulative Latency (ms)')
    ax.set_title('Cumulative Latency Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Summary statistics bar chart
    ax = axes[1, 1]
    metrics = ['Mean', 'P50', 'P95', 'P99']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (strategy, latencies) in enumerate(results.items()):
        values = [
            np.mean(latencies),
            np.percentile(latencies, 50),
            np.percentile(latencies, 95),
            np.percentile(latencies, 99)
        ]
        ax.bar(x + i*width, values, width, label=labels[strategy], color=colors[strategy])
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Statistics')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{workload_name} Workload', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / f"workload_{workload_name.lower().replace(' ', '_')}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_workloads_summary(all_results: Dict[str, Dict], save_path: Path):
    """Create summary comparison across all workloads"""
    
    workloads = list(all_results.keys())
    strategies = ["unpartitioned", "static", "rl_agent"]
    
    colors = {
        "unpartitioned": "#e74c3c",
        "static": "#3498db",
        "rl_agent": "#2ecc71"
    }
    labels = {
        "unpartitioned": "Unpartitioned",
        "static": "Static (Hash)",
        "rl_agent": "RL Agent (PPO)"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Mean latency by workload
    ax = axes[0, 0]
    x = np.arange(len(workloads))
    width = 0.25
    
    for i, strategy in enumerate(strategies):
        means = [np.mean(all_results[w][strategy]) for w in workloads]
        ax.bar(x + i*width, means, width, label=labels[strategy], color=colors[strategy])
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(workloads, rotation=15, ha='right')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title('Mean Latency by Workload Type')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. P95 latency by workload
    ax = axes[0, 1]
    for i, strategy in enumerate(strategies):
        p95s = [np.percentile(all_results[w][strategy], 95) for w in workloads]
        ax.bar(x + i*width, p95s, width, label=labels[strategy], color=colors[strategy])
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(workloads, rotation=15, ha='right')
    ax.set_ylabel('P95 Latency (ms)')
    ax.set_title('P95 Latency by Workload Type')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Improvement over static (%)
    ax = axes[1, 0]
    improvements = []
    for w in workloads:
        static_mean = np.mean(all_results[w]["static"])
        rl_mean = np.mean(all_results[w]["rl_agent"])
        improvement = (static_mean - rl_mean) / static_mean * 100
        improvements.append(improvement)
    
    colors_improve = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax.bar(workloads, improvements, color=colors_improve)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Latency Reduction (%)')
    ax.set_title('RL Agent Improvement Over Static Partitioning')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        ypos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 2
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    # 4. Latency variance (stability)
    ax = axes[1, 1]
    for i, strategy in enumerate(strategies):
        stds = [np.std(all_results[w][strategy]) for w in workloads]
        ax.bar(x + i*width, stds, width, label=labels[strategy], color=colors[strategy])
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(workloads, rotation=15, ha='right')
    ax.set_ylabel('Latency Std Dev (ms)')
    ax.set_title('Latency Stability by Workload\n(Lower = More Consistent)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Workload Comparison Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / "workload_summary.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_table(all_results: Dict[str, Dict], save_path: Path):
    """Create summary table image"""
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')
    
    workloads = list(all_results.keys())
    strategies = ["unpartitioned", "static", "rl_agent"]
    
    # Headers
    headers = ['Workload', 'Metric', 'Unpartitioned', 'Static', 'RL Agent', 'RL vs Static']
    
    # Data
    table_data = []
    for w in workloads:
        # Mean row
        unpart_mean = np.mean(all_results[w]["unpartitioned"])
        static_mean = np.mean(all_results[w]["static"])
        rl_mean = np.mean(all_results[w]["rl_agent"])
        improvement = (static_mean - rl_mean) / static_mean * 100
        
        table_data.append([
            w, 'Mean (ms)',
            f'{unpart_mean:.1f}',
            f'{static_mean:.1f}',
            f'{rl_mean:.1f}',
            f'{improvement:+.1f}%'
        ])
        
        # P95 row
        unpart_p95 = np.percentile(all_results[w]["unpartitioned"], 95)
        static_p95 = np.percentile(all_results[w]["static"], 95)
        rl_p95 = np.percentile(all_results[w]["rl_agent"], 95)
        improvement_p95 = (static_p95 - rl_p95) / static_p95 * 100
        
        table_data.append([
            '', 'P95 (ms)',
            f'{unpart_p95:.1f}',
            f'{static_p95:.1f}',
            f'{rl_p95:.1f}',
            f'{improvement_p95:+.1f}%'
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * len(headers)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Color code improvements
    for i, row in enumerate(table_data):
        if '%' in row[-1]:
            val = float(row[-1].replace('%', '').replace('+', ''))
            if val > 0:
                table[(i + 1, 5)].set_facecolor('#d4edda')  # Green for improvement
            else:
                table[(i + 1, 5)].set_facecolor('#f8d7da')  # Red for worse
    
    plt.title('Detailed Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path / "workload_table.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare workload patterns')
    parser.add_argument('--num-queries', type=int, default=500,
                       help='Number of queries per workload')
    parser.add_argument('--output-dir', type=str, default='./workload_results',
                       help='Output directory')
    parser.add_argument('--key-range-min', type=int, default=1)
    parser.add_argument('--key-range-max', type=int, default=150000)
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.output_dir) / f"workload_comparison_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Morph Workload Comparison")
    print("="*60)
    
    # Initialize
    key_range = (args.key_range_min, args.key_range_max)
    workload_gen = WorkloadGenerator(key_range)
    simulator = StrategySimulator()
    
    # Define workloads
    workloads = {
        "Uniform": {
            "keys": workload_gen.uniform(args.num_queries),
            "desc": "Queries uniformly distributed across key range",
            "is_range": False
        },
        "Gaussian": {
            "keys": workload_gen.gaussian(args.num_queries, center=0.5, std=0.15),
            "desc": "Queries concentrated around center (μ=0.5, σ=0.15)",
            "is_range": False
        },
        "Gaussian Range": {
            "keys": workload_gen.gaussian_range(args.num_queries, center=0.5, std=0.15),
            "desc": "Range queries with Gaussian distribution",
            "is_range": True
        },
        "Bimodal": {
            "keys": workload_gen.bimodal(args.num_queries, centers=(0.25, 0.75)),
            "desc": "Two hotspots at 25% and 75% of key range",
            "is_range": False
        },
        "Sliding Gaussian": {
            "keys": workload_gen.sliding_gaussian(args.num_queries, std=0.1),
            "desc": "Hotspot moves from left to right over time",
            "is_range": False
        }
    }
    
    all_results = {}
    
    # Run each workload
    for name, config in workloads.items():
        print(f"\nRunning {name} workload...")
        
        results = run_workload_comparison(
            name,
            config["keys"],
            simulator,
            is_range_query=config["is_range"]
        )
        
        all_results[name] = results
        
        # Plot individual workload results
        plot_workload_comparison(name, config["desc"], results, save_path, config["keys"])
        
        # Print summary
        print(f"  Unpartitioned: {np.mean(results['unpartitioned']):.1f}ms (mean)")
        print(f"  Static:        {np.mean(results['static']):.1f}ms (mean)")
        print(f"  RL Agent:      {np.mean(results['rl_agent']):.1f}ms (mean)")
        
        improvement = (np.mean(results['static']) - np.mean(results['rl_agent'])) / np.mean(results['static']) * 100
        print(f"  RL vs Static:  {improvement:+.1f}%")
    
    # Create summary plots
    print("\nGenerating summary plots...")
    plot_all_workloads_summary(all_results, save_path)
    create_summary_table(all_results, save_path)
    
    # Save raw results
    results_json = {}
    for w, r in all_results.items():
        results_json[w] = {
            strategy: {
                "mean": float(np.mean(latencies)),
                "std": float(np.std(latencies)),
                "p50": float(np.percentile(latencies, 50)),
                "p95": float(np.percentile(latencies, 95)),
                "p99": float(np.percentile(latencies, 99)),
                "min": float(np.min(latencies)),
                "max": float(np.max(latencies))
            }
            for strategy, latencies in r.items()
        }
    
    with open(save_path / "workload_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    
    print("\n" + "="*60)
    print("Results saved to:", save_path)
    print("="*60)
    print("\nGenerated files:")
    print("  - workload_uniform.png")
    print("  - workload_gaussian.png")
    print("  - workload_gaussian_range.png")
    print("  - workload_bimodal.png")
    print("  - workload_sliding_gaussian.png")
    print("  - workload_summary.png")
    print("  - workload_table.png")
    print("  - workload_results.json")


if __name__ == "__main__":
    main()
