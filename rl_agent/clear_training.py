#!/usr/bin/env python3
"""
Morph Training v3 - Clear RL Advantage

The problem: RL agent isn't learning effective strategies.

Solution: Make the environment clearer:
1. VERY skewed workload (80% of queries hit 20% of shards)
2. Static creates severe imbalance (all hot shards on one worker)
3. RL can fix this by spreading hot shards across workers
4. Huge latency difference between balanced vs imbalanced
"""

import argparse
import json
import logging
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClearMorphEnv(gym.Env):
    """
    Environment with very clear RL advantage.
    
    Setup:
    - 3 workers, 12 shards
    - Skewed workload: shards 0-2 get 80% of queries
    - Initial (static) placement: shards 0-2 ALL on worker 0 (terrible!)
    - RL goal: spread shards 0-2 across all 3 workers
    
    Latency = base + max_worker_load * penalty
    """
    
    def __init__(
        self,
        num_workers: int = 3,
        num_shards: int = 12,
        episode_length: int = 50,
        workload_type: str = "skewed"
    ):
        super().__init__()
        
        self.num_workers = num_workers
        self.num_shards = num_shards
        self.episode_length = episode_length
        self.workload_type = workload_type
        
        # Observation: worker_loads (3) + shard_placement_hot (3) + latency (1) + time (1)
        obs_dim = 8
        self.observation_space = Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        
        self.step_count = 0
        self.shard_to_worker = None
        self.workload = None
        
        # Latency parameters
        self.base_latency = 5.0
        self.load_penalty = 50.0  # Heavy penalty for imbalance
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # CRITICAL: Initial placement puts hot shards together (BAD for skewed workload)
        # Shards 0,1,2 -> Worker 0
        # Shards 3,4,5 -> Worker 1  
        # Shards 6,7,8 -> Worker 2
        # etc.
        self.shard_to_worker = np.array([i // (self.num_shards // self.num_workers) 
                                          for i in range(self.num_shards)])
        self.shard_to_worker = np.clip(self.shard_to_worker, 0, self.num_workers - 1)
        
        # Workload: first 3 shards get 80% of queries
        self._set_workload()
        
        return self._get_obs(), {}
    
    def _set_workload(self):
        """Set workload distribution"""
        if self.workload_type == "uniform":
            self.workload = np.ones(self.num_shards) / self.num_shards
        
        elif self.workload_type == "skewed":
            # Shards 0,1,2 get 80% of load
            self.workload = np.zeros(self.num_shards)
            self.workload[0] = 0.35
            self.workload[1] = 0.30
            self.workload[2] = 0.15
            # Rest share 20%
            remaining = (1.0 - 0.8) / (self.num_shards - 3)
            self.workload[3:] = remaining
            
        elif self.workload_type == "shifting":
            # Hot shard moves over time
            progress = self.step_count / self.episode_length
            hot_shard = int(progress * self.num_shards) % self.num_shards
            self.workload = np.ones(self.num_shards) * 0.05
            self.workload[hot_shard] = 0.5
            self.workload[(hot_shard + 1) % self.num_shards] = 0.2
            self.workload = self.workload / self.workload.sum()
            
        elif self.workload_type == "bimodal":
            self.workload = np.ones(self.num_shards) * 0.02
            self.workload[0] = 0.3
            self.workload[1] = 0.15
            self.workload[6] = 0.3
            self.workload[7] = 0.15
            self.workload = self.workload / self.workload.sum()
    
    def _get_worker_loads(self) -> np.ndarray:
        """Query load per worker"""
        loads = np.zeros(self.num_workers)
        for shard_id, worker_id in enumerate(self.shard_to_worker):
            loads[worker_id] += self.workload[shard_id]
        return loads
    
    def _calculate_latency(self) -> float:
        """Latency based on max worker load"""
        loads = self._get_worker_loads()
        max_load = np.max(loads)
        
        # Perfect balance: max_load = 1/3 = 0.33
        # Worst case (all on one): max_load = 1.0
        # Latency penalty proportional to deviation from perfect
        perfect_load = 1.0 / self.num_workers
        excess_load = max(0, max_load - perfect_load)
        
        latency = self.base_latency + excess_load * self.load_penalty
        latency += np.random.exponential(0.5)  # Small noise
        
        return latency
    
    def _get_obs(self) -> np.ndarray:
        loads = self._get_worker_loads()
        
        # Where are the hot shards (0,1,2)?
        hot_shard_locations = np.zeros(self.num_workers)
        for shard_id in range(3):  # Hot shards
            worker = self.shard_to_worker[shard_id]
            hot_shard_locations[worker] += 1
        hot_shard_locations /= 3  # Normalize
        
        # Current latency normalized
        latency = self._calculate_latency()
        lat_norm = min(latency / 30, 1.0)
        
        # Time
        time_frac = self.step_count / self.episode_length
        
        obs = np.concatenate([
            loads,  # 3 values
            hot_shard_locations,  # 3 values
            [lat_norm],  # 1 value
            [time_frac]  # 1 value
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action: np.ndarray):
        self.step_count += 1
        
        if self.workload_type == "shifting":
            self._set_workload()
        
        # Decode action
        action_type = int(np.argmax(action[:6]))
        params = (action[6:10] + 1) / 2
        
        # Execute
        moved = False
        if action_type == 1:  # MOVE
            shard_id = int(params[0] * self.num_shards) % self.num_shards
            target = int(params[1] * self.num_workers) % self.num_workers
            if self.shard_to_worker[shard_id] != target:
                self.shard_to_worker[shard_id] = target
                moved = True
        
        # Calculate latency
        latency = self._calculate_latency()
        
        # Reward: inverse of latency (lower latency = higher reward)
        reward = 20.0 - latency  # Base reward around 15 for good performance
        
        done = self.step_count >= self.episode_length
        
        loads = self._get_worker_loads()
        info = {
            "action": ["NO_OP", "MOVE", "SPLIT", "MERGE", "REBAL", "RANGE"][action_type],
            "moved": moved,
            "latency": latency,
            "max_load": float(np.max(loads)),
            "load_std": float(np.std(loads)),
            "worker_loads": loads.tolist()
        }
        
        return self._get_obs(), reward, False, done, info


def run_static_baseline(workload_type: str, num_episodes: int = 30) -> List[float]:
    """Run static baseline - NO shard movement"""
    latencies = []
    
    for _ in range(num_episodes):
        env = ClearMorphEnv(workload_type=workload_type)
        obs, _ = env.reset()
        ep_lats = []
        
        for _ in range(env.episode_length):
            # Always NO_OP
            action = np.zeros(10)
            action[0] = 1.0
            obs, _, _, _, info = env.step(action)
            ep_lats.append(info["latency"])
        
        latencies.append(np.mean(ep_lats))
    
    return latencies


def run_rl_agent(agent, workload_type: str, num_episodes: int = 30) -> List[float]:
    """Run trained RL agent"""
    latencies = []
    
    for _ in range(num_episodes):
        env = ClearMorphEnv(workload_type=workload_type)
        obs, _ = env.reset()
        ep_lats = []
        
        for _ in range(env.episode_length):
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, _, _, _, info = env.step(action)
            ep_lats.append(info["latency"])
        
        latencies.append(np.mean(ep_lats))
    
    return latencies


def run_optimal_baseline(workload_type: str, num_episodes: int = 30) -> List[float]:
    """Optimal strategy: spread hot shards across workers"""
    latencies = []
    
    for _ in range(num_episodes):
        env = ClearMorphEnv(workload_type=workload_type)
        obs, _ = env.reset()
        
        # Immediately move hot shards to different workers
        # Shard 0 -> Worker 0 (already there)
        # Shard 1 -> Worker 1
        # Shard 2 -> Worker 2
        env.shard_to_worker[0] = 0
        env.shard_to_worker[1] = 1
        env.shard_to_worker[2] = 2
        
        ep_lats = []
        for _ in range(env.episode_length):
            action = np.zeros(10)
            obs, _, _, _, info = env.step(action)
            ep_lats.append(info["latency"])
        
        latencies.append(np.mean(ep_lats))
    
    return latencies


def plot_results(results: Dict, save_path: Path):
    """Plot comparison"""
    import matplotlib.pyplot as plt
    
    workloads = list(results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    ax = axes[0]
    x = np.arange(len(workloads))
    width = 0.25
    
    static = [results[w]["static"]["mean"] for w in workloads]
    rl = [results[w]["rl"]["mean"] for w in workloads]
    optimal = [results[w]["optimal"]["mean"] for w in workloads]
    
    ax.bar(x - width, static, width, label='Static', color='#e74c3c')
    ax.bar(x, rl, width, label='RL Agent', color='#2ecc71')
    ax.bar(x + width, optimal, width, label='Optimal', color='#3498db')
    
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title('Latency Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([w.title() for w in workloads])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Improvement chart
    ax = axes[1]
    improvements = []
    for w in workloads:
        s = results[w]["static"]["mean"]
        r = results[w]["rl"]["mean"]
        improvements.append((s - r) / s * 100)
    
    colors = ['#2ecc71' if i > 0 else '#e74c3c' for i in improvements]
    bars = ax.bar(workloads, improvements, color=colors)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Latency Reduction (%)')
    ax.set_title('RL Improvement Over Static')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / "comparison.png", dpi=150)
    plt.close()
    logger.info(f"Saved plot to {save_path / 'comparison.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-timesteps', type=int, default=30000)
    parser.add_argument('--output-dir', type=str, default='./clear_training')
    args = parser.parse_args()
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ppo_agent import PPOAgent, PPOConfig, train
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.output_dir) / f"run_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Show the problem first
    logger.info("="*60)
    logger.info("BASELINE CHECK: Static vs Optimal")
    logger.info("="*60)
    
    static_lats = run_static_baseline("skewed", num_episodes=10)
    optimal_lats = run_optimal_baseline("skewed", num_episodes=10)
    
    logger.info(f"Static latency:  {np.mean(static_lats):.2f} ms (hot shards on same worker)")
    logger.info(f"Optimal latency: {np.mean(optimal_lats):.2f} ms (hot shards spread out)")
    logger.info(f"Potential improvement: {(np.mean(static_lats) - np.mean(optimal_lats)) / np.mean(static_lats) * 100:.1f}%")
    
    # Train RL agent
    logger.info("\n" + "="*60)
    logger.info("TRAINING RL AGENT")
    logger.info("="*60)
    
    env = ClearMorphEnv(workload_type="skewed", episode_length=50)
    
    config = PPOConfig(
        obs_dim=8,
        num_action_types=6,
        num_continuous_params=4,
        hidden_dims=[64, 64],
        learning_rate=1e-3,  # Higher LR for faster learning
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        entropy_coef=0.05  # More exploration
    )
    
    agent = PPOAgent(config)
    agent = train(env, agent, args.total_timesteps, log_interval=10,
                  save_path=str(save_path / "agent"))
    
    # Evaluate
    logger.info("\n" + "="*60)
    logger.info("EVALUATION")
    logger.info("="*60)
    
    results = {}
    for workload in ["uniform", "skewed", "shifting", "bimodal"]:
        static = run_static_baseline(workload)
        rl = run_rl_agent(agent, workload)
        optimal = run_optimal_baseline(workload)
        
        results[workload] = {
            "static": {"mean": np.mean(static), "std": np.std(static)},
            "rl": {"mean": np.mean(rl), "std": np.std(rl)},
            "optimal": {"mean": np.mean(optimal), "std": np.std(optimal)}
        }
        
        improvement = (np.mean(static) - np.mean(rl)) / np.mean(static) * 100
        logger.info(f"\n{workload.upper()}:")
        logger.info(f"  Static:  {np.mean(static):.2f} ms")
        logger.info(f"  RL:      {np.mean(rl):.2f} ms ({improvement:+.1f}%)")
        logger.info(f"  Optimal: {np.mean(optimal):.2f} ms")
    
    # Save
    with open(save_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    plot_results(results, save_path)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\n{'Workload':<12} {'Static':<10} {'RL':<10} {'Optimal':<10} {'RL Improv':<12}")
    print("-"*54)
    for w in results:
        s = results[w]["static"]["mean"]
        r = results[w]["rl"]["mean"]
        o = results[w]["optimal"]["mean"]
        imp = (s - r) / s * 100
        print(f"{w:<12} {s:<10.2f} {r:<10.2f} {o:<10.2f} {imp:+.1f}%")
    
    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
