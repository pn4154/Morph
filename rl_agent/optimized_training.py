#!/usr/bin/env python3
"""
Morph Optimized Training

Based on what worked best (final_training.py with 100k steps gave +60% on skewed).
This version:
1. Keeps the same working architecture
2. Trains longer (200k steps)
3. Focuses on the reward signal that worked
"""

import argparse
import json
import logging
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedMorphEnv(gym.Env):
    """Same env that gave +60% improvement"""
    
    def __init__(self, workload_type: str = "skewed", episode_length: int = 50, training: bool = True):
        super().__init__()
        
        self.num_workers = 3
        self.num_shards = 9
        self.episode_length = episode_length
        self.training = training
        self.workload_type = workload_type
        
        self.observation_space = Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        
        self.step_count = 0
        self.shard_to_worker = None
        self.workload = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # Random workload during training
        if self.training:
            self.workload_type = np.random.choice(
                ["skewed", "shifting", "bimodal"],
                p=[0.5, 0.3, 0.2]
            )
        
        # Bad initial: hot shards clustered on Worker 0
        self.shard_to_worker = np.array([
            0, 0, 0,
            1, 1, 1,
            2, 2, 2
        ])
        
        self._set_workload()
        return self._get_obs(), {"workload_type": self.workload_type}
    
    def _set_workload(self):
        if self.workload_type == "uniform":
            self.workload = np.ones(self.num_shards) / self.num_shards
        elif self.workload_type == "skewed":
            self.workload = np.array([0.35, 0.30, 0.15, 0.04, 0.04, 0.04, 0.04, 0.02, 0.02])
        elif self.workload_type == "shifting":
            self.workload = np.ones(self.num_shards) * 0.05
            hot_idx = (self.step_count // 10) % self.num_shards
            self.workload[hot_idx] = 0.55
            self.workload = self.workload / self.workload.sum()
        elif self.workload_type == "bimodal":
            self.workload = np.array([0.30, 0.10, 0.02, 0.02, 0.02, 0.30, 0.15, 0.05, 0.04])
    
    def _get_worker_loads(self):
        loads = np.zeros(self.num_workers)
        for shard_id, worker_id in enumerate(self.shard_to_worker):
            loads[worker_id] += self.workload[shard_id]
        return loads
    
    def _calculate_latency(self):
        loads = self._get_worker_loads()
        max_load = np.max(loads)
        ideal_load = 1.0 / self.num_workers
        latency = 5.0 + 50.0 * max(0, max_load - ideal_load)
        latency += np.random.normal(0, 0.3)
        return max(1.0, latency)
    
    def _get_obs(self):
        loads = self._get_worker_loads()
        time_frac = self.step_count / self.episode_length
        return np.array([loads[0], loads[1], loads[2], time_frac], dtype=np.float32)
    
    def step(self, action):
        self.step_count += 1
        
        if self.workload_type == "shifting":
            self._set_workload()
        
        action_type = int(np.argmax(action[:6]))
        
        if action_type == 1:  # MOVE
            shard_id = int((action[6] + 1) / 2 * self.num_shards) % self.num_shards
            target_worker = int((action[7] + 1) / 2 * self.num_workers) % self.num_workers
            if self.shard_to_worker[shard_id] != target_worker:
                self.shard_to_worker[shard_id] = target_worker
        
        latency = self._calculate_latency()
        reward = 30.0 - latency
        
        done = self.step_count >= self.episode_length
        return self._get_obs(), reward, False, done, {"latency": latency}


def evaluate(agent, workload_type: str, num_episodes: int = 50):
    """Evaluate with more episodes for stable results"""
    env = OptimizedMorphEnv(workload_type=workload_type, training=False)
    
    latencies = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        env.workload_type = workload_type
        env._set_workload()
        
        ep_lats = []
        for _ in range(env.episode_length):
            if agent is None:
                action = np.zeros(10)
                action[0] = 1.0
            else:
                action, _, _ = agent.select_action(obs, deterministic=True)
            obs, _, _, _, info = env.step(action)
            ep_lats.append(info["latency"])
        latencies.append(np.mean(ep_lats))
    
    return {"mean": np.mean(latencies), "std": np.std(latencies)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-timesteps', type=int, default=200000)
    parser.add_argument('--output-dir', type=str, default='./optimized_training')
    args = parser.parse_args()
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ppo_agent import PPOAgent, PPOConfig, train
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.output_dir) / f"run_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info(f"TRAINING FOR {args.total_timesteps} STEPS")
    logger.info("="*60)
    
    env = OptimizedMorphEnv(episode_length=50, training=True)
    
    config = PPOConfig(
        obs_dim=4,
        num_action_types=6,
        num_continuous_params=4,
        hidden_dims=[128, 128],
        learning_rate=5e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        entropy_coef=0.05,
    )
    
    agent = PPOAgent(config)
    agent = train(env, agent, args.total_timesteps, log_interval=50,
                  save_path=str(save_path / "agent"))
    
    # Evaluate with more episodes
    logger.info("\n" + "="*60)
    logger.info("EVALUATION (50 episodes each)")
    logger.info("="*60)
    
    results = {}
    workloads = ["uniform", "skewed", "shifting", "bimodal"]
    
    for workload in workloads:
        static = evaluate(None, workload, num_episodes=50)
        rl = evaluate(agent, workload, num_episodes=50)
        results[workload] = {"static": static, "rl": rl}
        
        imp = (static["mean"] - rl["mean"]) / static["mean"] * 100
        logger.info(f"{workload:12s}: Static={static['mean']:.2f}ms, RL={rl['mean']:.2f}ms ({imp:+.1f}%)")
    
    # Plot
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Latency comparison
    ax = axes[0]
    x = np.arange(len(workloads))
    width = 0.35
    
    static_vals = [results[w]["static"]["mean"] for w in workloads]
    rl_vals = [results[w]["rl"]["mean"] for w in workloads]
    
    ax.bar(x - width/2, static_vals, width, label='Static', color='#e74c3c')
    ax.bar(x + width/2, rl_vals, width, label='RL Agent', color='#2ecc71')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title('Latency Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([w.title() for w in workloads])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Improvement %
    ax = axes[1]
    improvements = [(results[w]["static"]["mean"] - results[w]["rl"]["mean"]) / 
                   results[w]["static"]["mean"] * 100 for w in workloads]
    colors = ['#27ae60' if i > 5 else '#f39c12' if i > 0 else '#e74c3c' for i in improvements]
    bars = ax.bar([w.title() for w in workloads], improvements, color=colors)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Improvement (%)')
    ax.set_title('RL Improvement Over Static')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontweight='bold')
    
    # 3. Speedup
    ax = axes[2]
    speedups = [results[w]["static"]["mean"] / max(results[w]["rl"]["mean"], 0.1) for w in workloads]
    colors = ['#27ae60' if s > 1.5 else '#f39c12' if s > 1 else '#e74c3c' for s in speedups]
    bars = ax.bar([w.title() for w in workloads], speedups, color=colors)
    ax.axhline(y=1, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Speedup (×)')
    ax.set_title('Speedup Factor')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}×', ha='center', fontweight='bold')
    
    plt.suptitle('Morph RL Agent Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / "results.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    total_s, total_r = 0, 0
    print(f"\n{'Workload':<12} {'Static':<12} {'RL':<12} {'Improvement':<12} {'Speedup':<10}")
    print("-"*58)
    
    for w in workloads:
        s = results[w]["static"]["mean"]
        r = results[w]["rl"]["mean"]
        imp = (s - r) / s * 100
        spd = s / r
        total_s += s
        total_r += r
        print(f"{w:<12} {s:<12.2f} {r:<12.2f} {imp:+.1f}%{'':<5} {spd:.2f}×")
    
    avg_imp = (total_s - total_r) / total_s * 100
    avg_spd = total_s / total_r
    print("-"*58)
    print(f"{'AVERAGE':<12} {total_s/4:<12.2f} {total_r/4:<12.2f} {avg_imp:+.1f}%{'':<5} {avg_spd:.2f}×")
    
    print(f"\nResults saved to: {save_path}")
    
    with open(save_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
