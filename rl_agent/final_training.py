#!/usr/bin/env python3
"""
Morph Final Training - Mixed Workloads

Trains on randomly sampled workloads for better generalization.
Longer training for better convergence.
"""

import argparse
import json
import logging
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from pathlib import Path
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MixedWorkloadEnv(gym.Env):
    """
    Environment that randomly samples workload type each episode.
    This trains a more robust agent.
    """
    
    def __init__(self, episode_length: int = 50, training: bool = True):
        super().__init__()
        
        self.num_workers = 3
        self.num_shards = 9
        self.episode_length = episode_length
        self.training = training
        
        # Will be set on reset
        self.workload_type = "skewed"
        
        self.observation_space = Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        
        self.step_count = 0
        self.shard_to_worker = None
        self.workload = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # Random workload type during training
        if self.training:
            self.workload_type = np.random.choice(["skewed", "shifting", "bimodal"], 
                                                   p=[0.5, 0.3, 0.2])
        
        # Bad initial placement: hot shards clustered
        self.shard_to_worker = np.array([
            0, 0, 0,  # Shards 0,1,2 -> Worker 0
            1, 1, 1,  # Shards 3,4,5 -> Worker 1
            2, 2, 2   # Shards 6,7,8 -> Worker 2
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
    
    def _get_worker_loads(self) -> np.ndarray:
        loads = np.zeros(self.num_workers)
        for shard_id, worker_id in enumerate(self.shard_to_worker):
            loads[worker_id] += self.workload[shard_id]
        return loads
    
    def _calculate_latency(self) -> float:
        loads = self._get_worker_loads()
        max_load = np.max(loads)
        ideal_load = 1.0 / self.num_workers
        
        latency = 5.0 + 50.0 * max(0, max_load - ideal_load)
        latency += np.random.normal(0, 0.3)
        return max(1.0, latency)
    
    def _get_obs(self) -> np.ndarray:
        loads = self._get_worker_loads()
        time_frac = self.step_count / self.episode_length
        return np.array([loads[0], loads[1], loads[2], time_frac], dtype=np.float32)
    
    def step(self, action: np.ndarray):
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


def evaluate(agent, workload_type: str, num_episodes: int = 30):
    """Evaluate agent on specific workload"""
    env = MixedWorkloadEnv(training=False)
    env.workload_type = workload_type
    
    latencies = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        env.workload_type = workload_type  # Override random selection
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
    
    return latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-timesteps', type=int, default=100000)
    parser.add_argument('--output-dir', type=str, default='./final_training')
    args = parser.parse_args()
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ppo_agent import PPOAgent, PPOConfig, train
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.output_dir) / f"run_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("TRAINING ON MIXED WORKLOADS")
    logger.info("="*60)
    
    env = MixedWorkloadEnv(episode_length=50, training=True)
    
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
    agent = train(env, agent, args.total_timesteps, log_interval=20,
                  save_path=str(save_path / "agent"))
    
    # Evaluate
    logger.info("\n" + "="*60)
    logger.info("EVALUATION")
    logger.info("="*60)
    
    results = {}
    workloads = ["uniform", "skewed", "shifting", "bimodal"]
    
    for workload in workloads:
        static_lats = evaluate(None, workload)
        rl_lats = evaluate(agent, workload)
        
        results[workload] = {
            "static": {"mean": np.mean(static_lats), "std": np.std(static_lats)},
            "rl": {"mean": np.mean(rl_lats), "std": np.std(rl_lats)}
        }
        
        improvement = (np.mean(static_lats) - np.mean(rl_lats)) / np.mean(static_lats) * 100
        logger.info(f"{workload:12s}: Static={np.mean(static_lats):.2f}ms, "
                   f"RL={np.mean(rl_lats):.2f}ms ({improvement:+.1f}%)")
    
    # Plot
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Latency comparison
    ax = axes[0, 0]
    x = np.arange(len(workloads))
    width = 0.35
    
    static_vals = [results[w]["static"]["mean"] for w in workloads]
    rl_vals = [results[w]["rl"]["mean"] for w in workloads]
    
    bars1 = ax.bar(x - width/2, static_vals, width, label='Static', color='#e74c3c')
    bars2 = ax.bar(x + width/2, rl_vals, width, label='RL Agent', color='#2ecc71')
    
    ax.set_ylabel('Mean Latency (ms)', fontsize=11)
    ax.set_title('Query Latency: Static vs RL Agent', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([w.title() for w in workloads])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', fontsize=9)
    
    # 2. Improvement percentage
    ax = axes[0, 1]
    improvements = [(results[w]["static"]["mean"] - results[w]["rl"]["mean"]) / 
                   results[w]["static"]["mean"] * 100 for w in workloads]
    colors = ['#2ecc71' if i > 0 else '#e74c3c' for i in improvements]
    bars = ax.bar([w.title() for w in workloads], improvements, color=colors, edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Latency Reduction (%)', fontsize=11)
    ax.set_title('RL Agent Improvement Over Static', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, improvements):
        ypos = bar.get_height() + 0.8 if val >= 0 else bar.get_height() - 2
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    # 3. Latency reduction in ms
    ax = axes[1, 0]
    reductions = [results[w]["static"]["mean"] - results[w]["rl"]["mean"] for w in workloads]
    colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in reductions]
    bars = ax.bar([w.title() for w in workloads], reductions, color=colors, edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Latency Saved (ms)', fontsize=11)
    ax.set_title('Absolute Latency Improvement', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, reductions):
        ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 1
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val:.1f}ms', ha='center', fontsize=10, fontweight='bold')
    
    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = []
    for w in workloads:
        s = results[w]["static"]["mean"]
        r = results[w]["rl"]["mean"]
        imp = (s - r) / s * 100
        table_data.append([w.title(), f'{s:.2f}', f'{r:.2f}', f'{imp:+.1f}%'])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Workload', 'Static (ms)', 'RL (ms)', 'Improvement'],
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0']*4
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Color code improvements
    for i, row in enumerate(table_data):
        val = float(row[3].replace('%', '').replace('+', ''))
        if val > 5:
            table[(i+1, 3)].set_facecolor('#d4edda')
        elif val > 0:
            table[(i+1, 3)].set_facecolor('#fff3cd')
        else:
            table[(i+1, 3)].set_facecolor('#f8d7da')
    
    ax.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle('Morph RL Agent: Database Partitioning Optimization', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path / "results.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS - MORPH RL AGENT")
    print("="*60)
    print(f"\n{'Workload':<12} {'Static (ms)':<14} {'RL (ms)':<12} {'Improvement':<12}")
    print("-"*50)
    
    total_static = 0
    total_rl = 0
    for w in workloads:
        s = results[w]["static"]["mean"]
        r = results[w]["rl"]["mean"]
        imp = (s - r) / s * 100
        total_static += s
        total_rl += r
        print(f"{w:<12} {s:<14.2f} {r:<12.2f} {imp:+.1f}%")
    
    avg_imp = (total_static - total_rl) / total_static * 100
    print("-"*50)
    print(f"{'AVERAGE':<12} {total_static/4:<14.2f} {total_rl/4:<12.2f} {avg_imp:+.1f}%")
    
    print(f"\nResults saved to: {save_path}")
    print(f"View results: open {save_path}/results.png")
    
    with open(save_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
