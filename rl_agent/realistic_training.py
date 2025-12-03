#!/usr/bin/env python3
"""
Improved Morph Training with Realistic Workload Simulation

This simulation models scenarios where RL CAN outperform static partitioning:
1. Skewed workloads - some shards get more queries than others
2. Time-varying hotspots - access patterns change over time
3. Load imbalance penalties - unbalanced shards cause higher latency

The key insight: Static hash partitioning is optimal for UNIFORM workloads.
RL should outperform when workloads are SKEWED or DYNAMIC.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealisticMorphEnv(gym.Env):
    """
    Realistic simulation where:
    - Workload has skew (some keys accessed more than others)
    - Hotspots shift over time
    - Shard imbalance causes latency penalties
    - Moving shards to match workload reduces latency
    """
    
    def __init__(
        self,
        num_workers: int = 3,
        num_shards: int = 18,
        episode_length: int = 100,
        workload_type: str = "skewed",  # uniform, skewed, shifting
        skew_factor: float = 0.8,  # Higher = more skewed
    ):
        super().__init__()
        
        self.num_workers = num_workers
        self.num_shards = num_shards
        self.episode_length = episode_length
        self.workload_type = workload_type
        self.skew_factor = skew_factor
        
        # Observation: shard_distribution(3) + workload_distribution(3) + latency_stats(4) + time(1) + imbalance(1)
        obs_dim = num_workers * 2 + 6
        self.observation_space = Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        
        # Action: [action_type(6), params(4)]
        self.action_space = Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        
        # State
        self.step_count = 0
        self.shard_to_worker = None  # Which worker each shard is on
        self.workload_distribution = None  # Query load per shard
        self.hotspot_center = None  # For shifting workloads
        self.recent_latencies = deque(maxlen=20)
        self.baseline_latency = 10.0  # Base query latency in ms
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        self.recent_latencies.clear()
        
        # Initialize shard placement (round-robin to workers)
        self.shard_to_worker = np.array([i % self.num_workers for i in range(self.num_shards)])
        
        # Initialize workload distribution based on type
        self._update_workload_distribution()
        
        # Initial hotspot for shifting workloads
        self.hotspot_center = 0.2
        
        return self._get_obs(), {"workload_type": self.workload_type}
    
    def _update_workload_distribution(self):
        """Update which shards are getting queries"""
        if self.workload_type == "uniform":
            # Equal load on all shards
            self.workload_distribution = np.ones(self.num_shards) / self.num_shards
            
        elif self.workload_type == "skewed":
            # Zipf-like distribution - few shards get most queries
            ranks = np.arange(1, self.num_shards + 1)
            weights = 1.0 / (ranks ** self.skew_factor)
            self.workload_distribution = weights / weights.sum()
            
        elif self.workload_type == "shifting":
            # Gaussian hotspot that moves over time
            progress = self.step_count / self.episode_length
            self.hotspot_center = 0.1 + progress * 0.8  # Move from 0.1 to 0.9
            
            shard_positions = np.linspace(0, 1, self.num_shards)
            weights = np.exp(-((shard_positions - self.hotspot_center) ** 2) / (2 * 0.1 ** 2))
            self.workload_distribution = weights / weights.sum()
            
        elif self.workload_type == "bimodal":
            # Two hotspots
            shard_positions = np.linspace(0, 1, self.num_shards)
            weights1 = np.exp(-((shard_positions - 0.25) ** 2) / (2 * 0.08 ** 2))
            weights2 = np.exp(-((shard_positions - 0.75) ** 2) / (2 * 0.08 ** 2))
            weights = weights1 + weights2
            self.workload_distribution = weights / weights.sum()
    
    def _get_worker_load(self) -> np.ndarray:
        """Calculate load on each worker based on shard placement and workload"""
        worker_load = np.zeros(self.num_workers)
        for shard_id, worker_id in enumerate(self.shard_to_worker):
            worker_load[worker_id] += self.workload_distribution[shard_id]
        return worker_load
    
    def _calculate_latency(self) -> float:
        """
        Calculate expected latency based on load balance.
        Key insight: Imbalanced load = higher latency on hot workers
        """
        worker_load = self._get_worker_load()
        
        # Base latency
        latency = self.baseline_latency
        
        # Imbalance penalty - higher variance in load = higher latency
        load_std = np.std(worker_load)
        load_mean = np.mean(worker_load)
        imbalance = load_std / (load_mean + 1e-6)
        
        # Imbalance causes up to 2x latency increase
        latency *= (1 + imbalance * 2)
        
        # Max load penalty - overloaded workers are slower
        max_load = np.max(worker_load)
        if max_load > 0.5:  # More than 50% of queries on one worker
            latency *= (1 + (max_load - 0.5) * 2)
        
        # Add noise
        latency += np.random.exponential(latency * 0.1)
        
        return latency
    
    def _get_obs(self) -> np.ndarray:
        """Build observation vector"""
        # Shard distribution per worker (normalized)
        shard_counts = np.zeros(self.num_workers)
        for worker_id in self.shard_to_worker:
            shard_counts[worker_id] += 1
        shard_dist = shard_counts / self.num_shards
        
        # Workload distribution per worker (normalized)
        worker_load = self._get_worker_load()
        
        # Latency stats
        if len(self.recent_latencies) > 0:
            latencies = list(self.recent_latencies)
            latency_stats = np.array([
                np.mean(latencies) / 100,
                np.percentile(latencies, 50) / 100,
                np.percentile(latencies, 95) / 100,
                np.std(latencies) / 50
            ])
        else:
            latency_stats = np.array([0.1, 0.1, 0.15, 0.05])
        
        # Time progress
        time_progress = self.step_count / self.episode_length
        
        # Load imbalance score
        imbalance = np.std(worker_load) / (np.mean(worker_load) + 1e-6)
        
        obs = np.concatenate([
            shard_dist,
            worker_load,
            np.clip(latency_stats, 0, 1),
            [time_progress],
            [min(imbalance, 1.0)]
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action: np.ndarray):
        self.step_count += 1
        
        # Update workload (for shifting patterns)
        if self.workload_type == "shifting":
            self._update_workload_distribution()
        
        # Decode action
        action_type = int(np.argmax(action[:6]))
        params = (action[6:10] + 1) / 2  # Normalize to [0, 1]
        
        # Execute action
        action_success, action_cost = self._execute_action(action_type, params)
        
        # Calculate latency after action
        latency = self._calculate_latency()
        self.recent_latencies.append(latency)
        
        # Calculate reward
        reward = self._calculate_reward(latency, action_cost, action_success)
        
        # Check termination
        done = self.step_count >= self.episode_length
        
        info = {
            "action_type": ["NO_OP", "MOVE", "SPLIT", "MERGE", "REBALANCE", "RANGE"][action_type],
            "action_success": action_success,
            "latency": latency,
            "worker_load": self._get_worker_load().tolist(),
            "imbalance": float(np.std(self._get_worker_load()))
        }
        
        return self._get_obs(), reward, False, done, info
    
    def _execute_action(self, action_type: int, params: np.ndarray) -> tuple:
        """Execute the action and return (success, cost)"""
        
        if action_type == 0:  # NO_OP
            return True, 0.0
        
        elif action_type == 1:  # MOVE_PARTITION
            # Move a shard to a different worker
            shard_id = int(params[0] * self.num_shards) % self.num_shards
            target_worker = int(params[1] * self.num_workers) % self.num_workers
            
            if self.shard_to_worker[shard_id] != target_worker:
                self.shard_to_worker[shard_id] = target_worker
                return True, 0.5  # Moving has a cost
            return True, 0.0
        
        elif action_type == 4:  # REBALANCE
            # Reset to round-robin (balanced)
            self.shard_to_worker = np.array([i % self.num_workers for i in range(self.num_shards)])
            return True, 1.0  # Rebalance is expensive
        
        else:  # SPLIT, MERGE, RANGE - not implemented
            return False, 0.0
    
    def _calculate_reward(self, latency: float, action_cost: float, action_success: bool) -> float:
        """
        Reward function:
        - Lower latency = higher reward
        - Balanced load = bonus
        - Action cost penalty
        - Failed action penalty
        """
        # Latency reward (target ~10ms baseline)
        latency_reward = (self.baseline_latency * 2 - latency) / self.baseline_latency
        latency_reward = np.clip(latency_reward, -2, 2)
        
        # Balance reward
        worker_load = self._get_worker_load()
        balance_score = 1.0 - np.std(worker_load) / (np.mean(worker_load) + 1e-6)
        balance_reward = balance_score * 0.5
        
        # Penalties
        cost_penalty = -action_cost * 0.2
        failure_penalty = -0.5 if not action_success else 0.0
        
        total = latency_reward + balance_reward + cost_penalty + failure_penalty
        return float(total)


def train_on_workload(workload_type: str, total_timesteps: int, save_path: Path):
    """Train an agent on a specific workload type"""
    
    from ppo_agent import PPOAgent, PPOConfig, train
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Training on {workload_type.upper()} workload")
    logger.info(f"{'='*50}")
    
    # Create environment
    env = RealisticMorphEnv(
        workload_type=workload_type,
        episode_length=100,
        skew_factor=0.8
    )
    
    # Create agent
    obs_dim = env.observation_space.shape[0]
    config = PPOConfig(
        obs_dim=obs_dim,
        num_action_types=6,
        num_continuous_params=4,
        hidden_dims=[128, 128],
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        total_timesteps=total_timesteps
    )
    
    agent = PPOAgent(config)
    
    # Train
    agent = train(
        env=env,
        agent=agent,
        total_timesteps=total_timesteps,
        log_interval=10,
        save_path=str(save_path / workload_type)
    )
    
    return agent, env


def evaluate_across_workloads(agents: dict, save_path: Path):
    """Evaluate all trained agents across all workload types"""
    
    workload_types = ["uniform", "skewed", "shifting", "bimodal"]
    results = {}
    
    for eval_workload in workload_types:
        logger.info(f"\nEvaluating on {eval_workload} workload...")
        
        env = RealisticMorphEnv(workload_type=eval_workload, episode_length=100)
        results[eval_workload] = {}
        
        for agent_name, agent in agents.items():
            episode_rewards = []
            episode_latencies = []
            
            for ep in range(10):
                obs, _ = env.reset()
                ep_reward = 0
                ep_latencies = []
                done = False
                
                while not done:
                    if agent is None:  # Static baseline
                        action = np.zeros(10)
                        action[0] = 1  # Always NO_OP
                    else:
                        action, _, _ = agent.select_action(obs, deterministic=True)
                    
                    obs, reward, _, done, info = env.step(action)
                    ep_reward += reward
                    ep_latencies.append(info['latency'])
                
                episode_rewards.append(ep_reward)
                episode_latencies.extend(ep_latencies)
            
            results[eval_workload][agent_name] = {
                "mean_reward": float(np.mean(episode_rewards)),
                "std_reward": float(np.std(episode_rewards)),
                "mean_latency": float(np.mean(episode_latencies)),
                "p95_latency": float(np.percentile(episode_latencies, 95))
            }
            
            logger.info(f"  {agent_name}: reward={np.mean(episode_rewards):.2f}, "
                       f"latency={np.mean(episode_latencies):.2f}ms")
    
    return results


def plot_results(results: dict, save_path: Path):
    """Plot comparison results"""
    import matplotlib.pyplot as plt
    
    workloads = list(results.keys())
    agents = list(results[workloads[0]].keys())
    
    colors = {
        "static": "#e74c3c",
        "rl_uniform": "#3498db",
        "rl_skewed": "#2ecc71",
        "rl_shifting": "#9b59b6",
        "rl_bimodal": "#f39c12"
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean Reward
    ax = axes[0]
    x = np.arange(len(workloads))
    width = 0.15
    
    for i, agent in enumerate(agents):
        rewards = [results[w][agent]["mean_reward"] for w in workloads]
        ax.bar(x + i*width, rewards, width, label=agent, 
               color=colors.get(agent, 'gray'))
    
    ax.set_xticks(x + width * (len(agents)-1) / 2)
    ax.set_xticklabels(workloads)
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Reward by Workload Type')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Mean Latency
    ax = axes[1]
    for i, agent in enumerate(agents):
        latencies = [results[w][agent]["mean_latency"] for w in workloads]
        ax.bar(x + i*width, latencies, width, label=agent,
               color=colors.get(agent, 'gray'))
    
    ax.set_xticks(x + width * (len(agents)-1) / 2)
    ax.set_xticklabels(workloads)
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title('Latency by Workload Type')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('RL Agent Performance Across Workload Types', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / "workload_comparison.png", dpi=150)
    plt.close()
    
    # Improvement over static
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, agent in enumerate(agents):
        if agent == "static":
            continue
        improvements = []
        for w in workloads:
            static_lat = results[w]["static"]["mean_latency"]
            agent_lat = results[w][agent]["mean_latency"]
            improvement = (static_lat - agent_lat) / static_lat * 100
            improvements.append(improvement)
        
        ax.bar(x + (i-1)*width, improvements, width, label=agent,
               color=colors.get(agent, 'gray'))
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.set_ylabel('Latency Reduction vs Static (%)')
    ax.set_title('RL Agent Improvement Over Static Partitioning')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path / "improvement_over_static.png", dpi=150)
    plt.close()
    
    logger.info(f"Plots saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train RL agents on realistic workloads')
    parser.add_argument('--total-timesteps', type=int, default=50000)
    parser.add_argument('--output-dir', type=str, default='./realistic_training')
    parser.add_argument('--eval-only', type=str, default=None,
                       help='Path to existing checkpoints for evaluation only')
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.output_dir) / f"run_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "rl_agent"))
    
    if args.eval_only:
        # Load existing agents
        logger.info(f"Loading agents from {args.eval_only}")
        from ppo_agent import PPOAgent, PPOConfig
        
        agents = {"static": None}
        for workload in ["uniform", "skewed", "shifting", "bimodal"]:
            checkpoint = Path(args.eval_only) / workload
            if checkpoint.exists():
                config = PPOConfig(obs_dim=12, num_action_types=6, num_continuous_params=4)
                agent = PPOAgent(config)
                agent.load(str(checkpoint))
                agents[f"rl_{workload}"] = agent
    else:
        # Train agents on different workloads
        agents = {"static": None}  # Static baseline (always NO_OP)
        
        for workload in ["skewed", "shifting"]:  # Train on challenging workloads
            agent, _ = train_on_workload(workload, args.total_timesteps, save_path)
            agents[f"rl_{workload}"] = agent
    
    # Evaluate across all workloads
    results = evaluate_across_workloads(agents, save_path)
    
    # Save results
    with open(save_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot
    plot_results(results, save_path)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for workload in results:
        print(f"\n{workload.upper()} workload:")
        static_lat = results[workload]["static"]["mean_latency"]
        for agent, metrics in results[workload].items():
            improvement = (static_lat - metrics["mean_latency"]) / static_lat * 100
            print(f"  {agent:15s}: latency={metrics['mean_latency']:6.2f}ms "
                  f"({improvement:+5.1f}% vs static)")
    
    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
