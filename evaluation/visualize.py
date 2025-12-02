#!/usr/bin/env python3
"""
Morph Visualization Module

Provides visualization utilities for training results,
evaluation metrics, and comparison analysis.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def set_style():
    """Set consistent plot style"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12


def plot_training_curves(
    checkpoint_path: str,
    output_path: str = 'training_curves.png'
):
    """Plot training curves from checkpoint"""
    import torch
    
    checkpoint = torch.load(
        Path(checkpoint_path) / 'agent.pt',
        map_location='cpu'
    )
    
    history = checkpoint.get('training_history', [])
    
    if not history:
        print("No training history found in checkpoint")
        return
    
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Policy loss
    axes[0, 0].plot([h['policy_loss'] for h in history], alpha=0.7)
    axes[0, 0].set_title('Policy Loss')
    axes[0, 0].set_xlabel('Update')
    axes[0, 0].set_ylabel('Loss')
    
    # Value loss
    axes[0, 1].plot([h['value_loss'] for h in history], alpha=0.7, color='orange')
    axes[0, 1].set_title('Value Loss')
    axes[0, 1].set_xlabel('Update')
    axes[0, 1].set_ylabel('Loss')
    
    # Entropy
    axes[0, 2].plot([h['entropy'] for h in history], alpha=0.7, color='green')
    axes[0, 2].set_title('Policy Entropy')
    axes[0, 2].set_xlabel('Update')
    axes[0, 2].set_ylabel('Entropy')
    
    # Approx KL
    axes[1, 0].plot([h['approx_kl'] for h in history], alpha=0.7, color='red')
    axes[1, 0].set_title('Approximate KL Divergence')
    axes[1, 0].set_xlabel('Update')
    axes[1, 0].set_ylabel('KL')
    
    # Clip fraction
    axes[1, 1].plot([h['clip_fraction'] for h in history], alpha=0.7, color='purple')
    axes[1, 1].set_title('Clip Fraction')
    axes[1, 1].set_xlabel('Update')
    axes[1, 1].set_ylabel('Fraction')
    
    # Combined metrics (normalized)
    ax = axes[1, 2]
    for metric in ['policy_loss', 'value_loss', 'entropy']:
        values = np.array([h[metric] for h in history])
        values = (values - values.min()) / (values.max() - values.min() + 1e-8)
        ax.plot(values, alpha=0.7, label=metric)
    ax.set_title('Normalized Metrics')
    ax.set_xlabel('Update')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {output_path}")


def plot_episode_rewards(
    rewards: List[float],
    output_path: str = 'episode_rewards.png',
    window: int = 10
):
    """Plot episode rewards with moving average"""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Raw rewards
    ax.plot(rewards, alpha=0.3, label='Episode Reward')
    
    # Moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), moving_avg, 
                linewidth=2, label=f'Moving Avg ({window})')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Episode Rewards')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Episode rewards plot saved to {output_path}")


def plot_baseline_comparison(
    results_path: str,
    output_path: str = 'baseline_comparison.png'
):
    """Plot comparison of baseline strategies"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    baselines = results.get('baselines', {})
    
    if not baselines:
        print("No baseline results found")
        return
    
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    baseline_names = list(baselines.keys())
    workload_types = ['oltp', 'olap', 'mixed']
    colors = sns.color_palette("husl", len(baseline_names))
    
    # Mean Latency Comparison
    ax = axes[0, 0]
    x = np.arange(len(workload_types))
    width = 0.25
    
    for i, baseline in enumerate(baseline_names):
        latencies = []
        for wl in workload_types:
            if wl in baselines[baseline].get('workloads', {}):
                latencies.append(baselines[baseline]['workloads'][wl]['mean_latency_ms'])
            else:
                latencies.append(0)
        ax.bar(x + i*width, latencies, width, label=baseline, color=colors[i])
    
    ax.set_xlabel('Workload Type')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title('Mean Latency by Strategy')
    ax.set_xticks(x + width)
    ax.set_xticklabels(workload_types)
    ax.legend()
    
    # P95 Latency Comparison
    ax = axes[0, 1]
    for i, baseline in enumerate(baseline_names):
        p95_latencies = []
        for wl in workload_types:
            if wl in baselines[baseline].get('workloads', {}):
                p95_latencies.append(baselines[baseline]['workloads'][wl]['p95_latency_ms'])
            else:
                p95_latencies.append(0)
        ax.bar(x + i*width, p95_latencies, width, label=baseline, color=colors[i])
    
    ax.set_xlabel('Workload Type')
    ax.set_ylabel('P95 Latency (ms)')
    ax.set_title('P95 Latency by Strategy')
    ax.set_xticks(x + width)
    ax.set_xticklabels(workload_types)
    ax.legend()
    
    # Throughput Comparison
    ax = axes[1, 0]
    for i, baseline in enumerate(baseline_names):
        throughputs = []
        for wl in workload_types:
            if wl in baselines[baseline].get('workloads', {}):
                throughputs.append(baselines[baseline]['workloads'][wl]['throughput_qps'])
            else:
                throughputs.append(0)
        ax.bar(x + i*width, throughputs, width, label=baseline, color=colors[i])
    
    ax.set_xlabel('Workload Type')
    ax.set_ylabel('Throughput (QPS)')
    ax.set_title('Throughput by Strategy')
    ax.set_xticks(x + width)
    ax.set_xticklabels(workload_types)
    ax.legend()
    
    # Shard Distribution (for first baseline with partition metrics)
    ax = axes[1, 1]
    for baseline in baseline_names:
        if 'partition_metrics' in baselines[baseline]:
            shards = baselines[baseline]['partition_metrics'].get('shards_per_node', {})
            if shards:
                nodes = list(shards.keys())
                counts = list(shards.values())
                ax.bar(nodes, counts, alpha=0.7, label=baseline)
                break
    
    ax.set_xlabel('Node')
    ax.set_ylabel('Shard Count')
    ax.set_title('Shard Distribution')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Baseline comparison plot saved to {output_path}")


def plot_action_distribution(
    results_path: str,
    output_path: str = 'action_distribution.png'
):
    """Plot distribution of actions taken by the agent"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    action_dist = results.get('action_distribution', {})
    
    if not action_dist:
        print("No action distribution found")
        return
    
    set_style()
    
    action_names = [
        'NO_OP', 'MOVE_PARTITION', 'SPLIT_PARTITION',
        'MERGE_PARTITIONS', 'REBALANCE_PARTITION', 'MOVE_DATA_RANGE'
    ]
    
    counts = [action_dist.get(str(i), 0) for i in range(6)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", 6)
    bars = ax.bar(action_names, counts, color=colors)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Action Type')
    ax.set_ylabel('Count')
    ax.set_title('Agent Action Distribution')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Action distribution plot saved to {output_path}")


def plot_latency_over_time(
    latencies: List[float],
    output_path: str = 'latency_over_time.png',
    window: int = 50
):
    """Plot latency measurements over time"""
    set_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time series
    ax = axes[0]
    ax.plot(latencies, alpha=0.5, linewidth=0.5, label='Raw')
    
    if len(latencies) >= window:
        moving_avg = np.convolve(latencies, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(latencies)), moving_avg, 
                linewidth=2, color='red', label=f'Moving Avg ({window})')
    
    ax.set_xlabel('Query Number')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Query Latency Over Time')
    ax.legend()
    
    # Distribution
    ax = axes[1]
    ax.hist(latencies, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(latencies), color='red', linestyle='--', 
               label=f'Mean: {np.mean(latencies):.2f}ms')
    ax.axvline(np.percentile(latencies, 95), color='orange', linestyle='--',
               label=f'P95: {np.percentile(latencies, 95):.2f}ms')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Latency Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Latency plot saved to {output_path}")


def generate_report(
    checkpoint_path: str,
    eval_results_path: Optional[str] = None,
    output_dir: str = './reports'
):
    """Generate comprehensive visualization report"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualization report...")
    
    # Training curves
    if Path(checkpoint_path).exists():
        plot_training_curves(
            checkpoint_path,
            str(output_path / 'training_curves.png')
        )
    
    # Evaluation results
    if eval_results_path and Path(eval_results_path).exists():
        plot_baseline_comparison(
            eval_results_path,
            str(output_path / 'baseline_comparison.png')
        )
        
        with open(eval_results_path, 'r') as f:
            results = json.load(f)
        
        if 'action_distribution' in results:
            plot_action_distribution(
                eval_results_path,
                str(output_path / 'action_distribution.png')
            )
    
    print(f"\nReport generated in {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Morph Visualization Tools')
    parser.add_argument('--checkpoint', type=str, help='Path to training checkpoint')
    parser.add_argument('--eval-results', type=str, help='Path to evaluation results JSON')
    parser.add_argument('--output-dir', type=str, default='./reports',
                       help='Output directory for plots')
    parser.add_argument('--plot-type', type=str, 
                       choices=['training', 'baseline', 'actions', 'all'],
                       default='all', help='Type of plot to generate')
    
    args = parser.parse_args()
    
    if args.plot_type == 'all':
        generate_report(
            args.checkpoint or './checkpoints',
            args.eval_results,
            args.output_dir
        )
    elif args.plot_type == 'training' and args.checkpoint:
        plot_training_curves(args.checkpoint, f'{args.output_dir}/training.png')
    elif args.plot_type == 'baseline' and args.eval_results:
        plot_baseline_comparison(args.eval_results, f'{args.output_dir}/baseline.png')
    elif args.plot_type == 'actions' and args.eval_results:
        plot_action_distribution(args.eval_results, f'{args.output_dir}/actions.png')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
