#!/usr/bin/env python3
"""
Morph Training v2 - Improved Simulation

Key fix: Static partitioning should NOT adapt to workload changes.
The previous simulation accidentally let static benefit from the same
balanced initial state as RL.

This version properly models:
1. Static: Fixed hash distribution, CANNOT adapt to hotspots
2. RL: Can move shards to balance load

The key insight: Static partitioning assigns shards randomly by hash,
which creates IMBALANCE when workload is skewed.
"""

import argparse
import json
import logging
import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import Box
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedMorphEnv(gym.Env):
    """
    Environment where RL can genuinely outperform static partitioning.
    
    Key mechanics:
    - Workload hits certain shards more than others (skew)
    - Shards are initially distributed by hash (may be imbalanced for workload)
    - Static cannot adapt; RL can move shards to reduce load imbalance
    - Latency = base + (load_on_busiest_worker - average_load) * penalty
    """
    
    def __init__(
        self,
        num_workers: int = 3,
        num_shards: int = 18,
        episode_length: int = 100,
        workload_type: str = "skewed",
        skew_factor: float = 1.5,  # Zipf exponent - higher = more skewed
    ):
        super().__init__()
        
        self.num_workers = num_workers
        self.num_shards = num_shards
        self.episode_length = episode_length
        self.workload_type = workload_type
        self.skew_factor = skew_factor
        
        # Observation space
        obs_dim = num_workers * 2 + 6  # shard_dist + load_dist + stats
        self.observation_space = Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        
        # State
        self.step_count = 0
        self.shard_to_worker = None
        self.workload_per_shard = None
        self.recent_latencies = deque(maxlen=50)
        
        # Constants
        self.base_latency = 10.0
        self.imbalance_penalty = 20.0  # ms penalty per unit of imbalance
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        self.recent_latencies.clear()
        
        # Hash-based shard assignment (what Citus does)
        # This is FIXED for static, but RL can modify it
        self.shard_to_worker = np.array([i % self.num_workers for i in range(self.num_shards)])
        
        # Generate workload distribution
        self._generate_workload()
        
        return self._get_obs(), {}
    
    def _generate_workload(self):
        """Generate skewed workload - some shards get way more queries"""
        if self.workload_type == "uniform":
            self.workload_per_shard = np.ones(self.num_shards) / self.num_shards
            
        elif self.workload_type == "skewed":
            # Zipf distribution - first few shards get most queries
            ranks = np.arange(1, self.num_shards + 1)
            weights = 1.0 / (ranks ** self.skew_factor)
            # Shuffle so hot shards aren't always at the start
            np.random.shuffle(weights)
            self.workload_per_shard = weights / weights.sum()
            
        elif self.workload_type == "shifting":
            # Hotspot moves over time
            progress = self.step_count / self.episode_length
            hotspot_shard = int(progress * self.num_shards) % self.num_shards
            
            weights = np.ones(self.num_shards) * 0.1
            # Hot region around hotspot
            for offset in range(-2, 3):
                idx = (hotspot_shard + offset) % self.num_shards
                weights[idx] = 2.0 - abs(offset) * 0.3
            self.workload_per_shard = weights / weights.sum()
            
        elif self.workload_type == "bimodal":
            weights = np.ones(self.num_shards) * 0.1
            # Two hotspots
            hot1 = self.num_shards // 4
            hot2 = 3 * self.num_shards // 4
            for offset in range(-1, 2):
                weights[(hot1 + offset) % self.num_shards] = 2.0
                weights[(hot2 + offset) % self.num_shards] = 2.0
            self.workload_per_shard = weights / weights.sum()
    
    def _get_worker_loads(self) -> np.ndarray:
        """Calculate query load on each worker"""
        loads = np.zeros(self.num_workers)
        for shard_id, worker_id in enumerate(self.shard_to_worker):
            loads[worker_id] += self.workload_per_shard[shard_id]
        return loads
    
    def _calculate_latency(self) -> Tuple[float, dict]:
        """
        Latency model:
        - Base latency for all queries
        - PLUS penalty proportional to load imbalance
        - The busiest worker becomes a bottleneck
        """
        loads = self._get_worker_loads()
        
        max_load = np.max(loads)
        avg_load = np.mean(loads)
        imbalance = max_load - avg_load  # How much more the busiest worker has
        
        # Latency increases with imbalance
        latency = self.base_latency + imbalance * self.imbalance_penalty
        
        # Add some noise
        latency += np.random.exponential(1.0)
        
        metrics = {
            "max_load": max_load,
            "avg_load": avg_load,
            "imbalance": imbalance,
            "worker_loads": loads.tolist()
        }
        
        return latency, metrics
    
    def _get_obs(self) -> np.ndarray:
        """Build observation"""
        # Shard distribution (how many shards per worker)
        shard_counts = np.zeros(self.num_workers)
        for w in self.shard_to_worker:
            shard_counts[w] += 1
        shard_dist = shard_counts / self.num_shards
        
        # Load distribution (query load per worker)
        loads = self._get_worker_loads()
        
        # Latency stats
        if len(self.recent_latencies) > 0:
            lats = list(self.recent_latencies)
            lat_stats = np.array([
                np.mean(lats) / 50,
                np.std(lats) / 20,
                np.max(lats) / 100,
                np.min(lats) / 50
            ])
        else:
            lat_stats = np.array([0.2, 0.1, 0.3, 0.1])
        
        # Time and imbalance
        time_frac = self.step_count / self.episode_length
        imbalance = np.max(loads) - np.mean(loads)
        
        obs = np.concatenate([
            shard_dist,
            loads,
            np.clip(lat_stats, 0, 1),
            [time_frac, min(imbalance * 2, 1.0)]
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action: np.ndarray):
        self.step_count += 1
        
        # Update workload for shifting patterns
        if self.workload_type == "shifting":
            self._generate_workload()
        
        # Decode and execute action
        action_type = int(np.argmax(action[:6]))
        params = (action[6:10] + 1) / 2
        
        action_success, action_cost = self._execute_action(action_type, params)
        
        # Measure latency
        latency, metrics = self._calculate_latency()
        self.recent_latencies.append(latency)
        
        # Reward: lower latency = higher reward
        # Target latency is base_latency (perfect balance)
        latency_reward = (self.base_latency * 2 - latency) / self.base_latency
        
        # Small penalty for taking actions (encourages efficiency)
        action_penalty = action_cost * 0.1
        
        reward = latency_reward - action_penalty
        
        done = self.step_count >= self.episode_length
        
        info = {
            "action_type": ["NO_OP", "MOVE", "SPLIT", "MERGE", "REBALANCE", "RANGE"][action_type],
            "action_success": action_success,
            "latency": latency,
            "imbalance": metrics["imbalance"],
            "worker_loads": metrics["worker_loads"]
        }
        
        return self._get_obs(), reward, False, done, info
    
    def _execute_action(self, action_type: int, params: np.ndarray) -> Tuple[bool, float]:
        """Execute action"""
        if action_type == 0:  # NO_OP
            return True, 0.0
        
        elif action_type == 1:  # MOVE_PARTITION
            shard_id = int(params[0] * self.num_shards) % self.num_shards
            target_worker = int(params[1] * self.num_workers) % self.num_workers
            
            if self.shard_to_worker[shard_id] != target_worker:
                self.shard_to_worker[shard_id] = target_worker
                return True, 0.3
            return True, 0.0
        
        elif action_type == 4:  # REBALANCE - redistribute evenly
            self.shard_to_worker = np.array([i % self.num_workers for i in range(self.num_shards)])
            return True, 0.5
        
        return False, 0.0


class StaticEnv(ImprovedMorphEnv):
    """
    Static environment - NO actions are executed.
    This represents what happens with default Citus hash partitioning.
    """
    
    def step(self, action: np.ndarray):
        self.step_count += 1
        
        if self.workload_type == "shifting":
            self._generate_workload()
        
        # NO ACTION - static partitioning
        latency, metrics = self._calculate_latency()
        self.recent_latencies.append(latency)
        
        latency_reward = (self.base_latency * 2 - latency) / self.base_latency
        reward = latency_reward
        
        done = self.step_count >= self.episode_length
        
        info = {
            "action_type": "STATIC",
            "latency": latency,
            "imbalance": metrics["imbalance"]
        }
        
        return self._get_obs(), reward, False, done, info


def evaluate_strategies(
    workload_type: str,
    rl_agent,
    num_episodes: int = 20
) -> Dict[str, Dict]:
    """Compare RL agent vs Static on a workload"""
    
    results = {"rl_agent": [], "static": []}
    
    # RL Agent
    env = ImprovedMorphEnv(workload_type=workload_type, episode_length=100)
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_latencies = []
        done = False
        while not done:
            action, _, _ = rl_agent.select_action(obs, deterministic=True)
            obs, _, _, done, info = env.step(action)
            ep_latencies.append(info["latency"])
        results["rl_agent"].append(np.mean(ep_latencies))
    
    # Static (no adaptation)
    env = StaticEnv(workload_type=workload_type, episode_length=100)
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_latencies = []
        done = False
        while not done:
            obs, _, _, done, info = env.step(np.zeros(10))  # Action ignored
            ep_latencies.append(info["latency"])
        results["static"].append(np.mean(ep_latencies))
    
    return {
        "rl_agent": {
            "mean_latency": float(np.mean(results["rl_agent"])),
            "std_latency": float(np.std(results["rl_agent"]))
        },
        "static": {
            "mean_latency": float(np.mean(results["static"])),
            "std_latency": float(np.std(results["static"]))
        }
    }


def plot_comparison(all_results: Dict, save_path: Path):
    """Generate comparison plots"""
    import matplotlib.pyplot as plt
    
    workloads = list(all_results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Latency comparison
    ax = axes[0]
    x = np.arange(len(workloads))
    width = 0.35
    
    rl_latencies = [all_results[w]["rl_agent"]["mean_latency"] for w in workloads]
    static_latencies = [all_results[w]["static"]["mean_latency"] for w in workloads]
    
    bars1 = ax.bar(x - width/2, static_latencies, width, label='Static (Hash)', color='#e74c3c')
    bars2 = ax.bar(x + width/2, rl_latencies, width, label='RL Agent', color='#2ecc71')
    
    ax.set_ylabel('Mean Latency (ms)', fontsize=12)
    ax.set_xlabel('Workload Type', fontsize=12)
    ax.set_title('Latency: RL Agent vs Static Partitioning', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([w.title() for w in workloads])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', fontsize=9)
    
    # Improvement percentage
    ax = axes[1]
    improvements = []
    for w in workloads:
        static_lat = all_results[w]["static"]["mean_latency"]
        rl_lat = all_results[w]["rl_agent"]["mean_latency"]
        improvement = (static_lat - rl_lat) / static_lat * 100
        improvements.append(improvement)
    
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax.bar(workloads, improvements, color=colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Latency Reduction (%)', fontsize=12)
    ax.set_xlabel('Workload Type', fontsize=12)
    ax.set_title('RL Agent Improvement Over Static', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, improvements):
        ypos = bar.get_height() + 1 if val >= 0 else bar.get_height() - 3
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / "rl_vs_static_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Learning curve for reference
    logger.info(f"Saved comparison plot to {save_path / 'rl_vs_static_comparison.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-timesteps', type=int, default=50000)
    parser.add_argument('--output-dir', type=str, default='./improved_training')
    args = parser.parse_args()
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ppo_agent import PPOAgent, PPOConfig, train
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.output_dir) / f"run_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Train on skewed workload (where RL should help most)
    logger.info("="*60)
    logger.info("Training RL Agent on SKEWED workload")
    logger.info("="*60)
    
    env = ImprovedMorphEnv(workload_type="skewed", episode_length=100, skew_factor=1.5)
    
    config = PPOConfig(
        obs_dim=env.observation_space.shape[0],
        num_action_types=6,
        num_continuous_params=4,
        hidden_dims=[128, 128],
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10
    )
    
    agent = PPOAgent(config)
    agent = train(env, agent, args.total_timesteps, log_interval=10, 
                  save_path=str(save_path / "agent"))
    
    # Evaluate on all workload types
    logger.info("\n" + "="*60)
    logger.info("Evaluating RL vs Static across workloads")
    logger.info("="*60)
    
    all_results = {}
    for workload in ["uniform", "skewed", "shifting", "bimodal"]:
        logger.info(f"\nEvaluating on {workload} workload...")
        results = evaluate_strategies(workload, agent, num_episodes=20)
        all_results[workload] = results
        
        static_lat = results["static"]["mean_latency"]
        rl_lat = results["rl_agent"]["mean_latency"]
        improvement = (static_lat - rl_lat) / static_lat * 100
        
        logger.info(f"  Static:   {static_lat:.2f} ms")
        logger.info(f"  RL Agent: {rl_lat:.2f} ms")
        logger.info(f"  Improvement: {improvement:+.1f}%")
    
    # Save and plot
    with open(save_path / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    plot_comparison(all_results, save_path)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\n{'Workload':<12} {'Static (ms)':<14} {'RL (ms)':<12} {'Improvement':<12}")
    print("-"*50)
    for w in all_results:
        s = all_results[w]["static"]["mean_latency"]
        r = all_results[w]["rl_agent"]["mean_latency"]
        imp = (s - r) / s * 100
        print(f"{w:<12} {s:<14.2f} {r:<12.2f} {imp:+.1f}%")
    
    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
