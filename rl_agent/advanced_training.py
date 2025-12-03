#!/usr/bin/env python3
"""
Morph Advanced Training - Maximum Performance

Improvements:
1. Larger network (256x256)
2. Longer training (200k steps)
3. Better reward shaping
4. More aggressive shard movement
5. Train separate specialists for each workload
6. Ensemble the best strategies
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
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedMorphEnv(gym.Env):
    """
    Advanced environment with:
    - Richer observation (includes per-shard info)
    - Better reward shaping
    - More realistic latency model
    """
    
    def __init__(self, workload_type: str = "skewed", episode_length: int = 100):
        super().__init__()
        
        self.num_workers = 3
        self.num_shards = 9
        self.episode_length = episode_length
        self.workload_type = workload_type
        
        # Richer observation:
        # - worker loads (3)
        # - hot shard locations (one-hot for each hot shard) (9)
        # - current max imbalance (1)
        # - time (1)
        # Total: 14
        obs_dim = 14
        self.observation_space = Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        
        self.step_count = 0
        self.shard_to_worker = None
        self.workload = None
        self.prev_latency = None
        self.best_latency = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # Bad initial: cluster hot shards
        self.shard_to_worker = np.array([
            0, 0, 0,  # Hot shards on Worker 0
            1, 1, 1,
            2, 2, 2
        ])
        
        self._set_workload()
        self.prev_latency = self._calculate_latency()
        self.best_latency = self.prev_latency
        
        return self._get_obs(), {}
    
    def _set_workload(self):
        if self.workload_type == "uniform":
            self.workload = np.ones(self.num_shards) / self.num_shards
        
        elif self.workload_type == "skewed":
            # Very skewed: 85% on 3 shards
            self.workload = np.array([0.40, 0.30, 0.15, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02])
        
        elif self.workload_type == "shifting":
            progress = self.step_count / self.episode_length
            hot_idx = int(progress * self.num_shards) % self.num_shards
            self.workload = np.ones(self.num_shards) * 0.03
            self.workload[hot_idx] = 0.50
            self.workload[(hot_idx + 1) % self.num_shards] = 0.20
            self.workload = self.workload / self.workload.sum()
        
        elif self.workload_type == "bimodal":
            # Two clear hotspots
            self.workload = np.array([0.35, 0.10, 0.02, 0.02, 0.02, 0.35, 0.10, 0.02, 0.02])
    
    def _get_worker_loads(self) -> np.ndarray:
        loads = np.zeros(self.num_workers)
        for shard_id, worker_id in enumerate(self.shard_to_worker):
            loads[worker_id] += self.workload[shard_id]
        return loads
    
    def _calculate_latency(self) -> float:
        loads = self._get_worker_loads()
        max_load = np.max(loads)
        ideal_load = 1.0 / self.num_workers
        
        # More aggressive penalty
        base = 5.0
        penalty = 60.0
        
        latency = base + penalty * max(0, max_load - ideal_load)
        return latency
    
    def _get_obs(self) -> np.ndarray:
        loads = self._get_worker_loads()
        
        # One-hot encoding of shard locations
        shard_locs = np.zeros(self.num_shards)
        for i in range(self.num_shards):
            shard_locs[i] = self.shard_to_worker[i] / (self.num_workers - 1)
        
        # Imbalance metric
        imbalance = np.std(loads) / (np.mean(loads) + 1e-6)
        
        time_frac = self.step_count / self.episode_length
        
        obs = np.concatenate([
            loads,
            shard_locs,
            [min(imbalance, 1.0)],
            [time_frac]
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action: np.ndarray):
        self.step_count += 1
        
        if self.workload_type == "shifting":
            self._set_workload()
        
        # Decode action
        action_type = int(np.argmax(action[:6]))
        
        moved = False
        if action_type == 1:  # MOVE
            shard_id = int((action[6] + 1) / 2 * self.num_shards) % self.num_shards
            target_worker = int((action[7] + 1) / 2 * self.num_workers) % self.num_workers
            
            if self.shard_to_worker[shard_id] != target_worker:
                self.shard_to_worker[shard_id] = target_worker
                moved = True
        
        latency = self._calculate_latency()
        
        # Improved reward shaping:
        # 1. Base reward for low latency
        # 2. Bonus for improvement over previous
        # 3. Bonus for beating best so far
        
        base_reward = 35.0 - latency
        
        improvement_bonus = 0
        if self.prev_latency is not None:
            improvement = self.prev_latency - latency
            improvement_bonus = improvement * 2  # Reward improvements
        
        best_bonus = 0
        if latency < self.best_latency:
            best_bonus = 5.0
            self.best_latency = latency
        
        reward = base_reward + improvement_bonus + best_bonus
        
        self.prev_latency = latency
        
        done = self.step_count >= self.episode_length
        
        return self._get_obs(), reward, False, done, {
            "latency": latency,
            "moved": moved,
            "loads": self._get_worker_loads().tolist()
        }


def train_specialist(workload_type: str, total_timesteps: int, save_path: Path):
    """Train a specialist agent for one workload type"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ppo_agent import PPOAgent, PPOConfig, train
    
    logger.info(f"\nTraining {workload_type.upper()} specialist...")
    
    env = AdvancedMorphEnv(workload_type=workload_type, episode_length=100)
    
    config = PPOConfig(
        obs_dim=14,
        num_action_types=6,
        num_continuous_params=4,
        hidden_dims=[256, 256],  # Larger network
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=15,  # More epochs
        gamma=0.995,  # Higher gamma for long-term planning
        gae_lambda=0.98,
        entropy_coef=0.02,
    )
    
    agent = PPOAgent(config)
    agent = train(env, agent, total_timesteps, log_interval=20,
                  save_path=str(save_path / f"specialist_{workload_type}"))
    
    return agent


def evaluate_agent(agent, workload_type: str, num_episodes: int = 30) -> Dict:
    """Evaluate agent on a workload"""
    env = AdvancedMorphEnv(workload_type=workload_type, episode_length=100)
    
    latencies = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
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
    
    return {
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies))
    }


def plot_results(results: Dict, save_path: Path):
    """Generate comprehensive results visualization"""
    import matplotlib.pyplot as plt
    
    workloads = list(results.keys())
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Main comparison bar chart
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(len(workloads))
    width = 0.35
    
    static = [results[w]["static"]["mean"] for w in workloads]
    rl = [results[w]["rl"]["mean"] for w in workloads]
    
    bars1 = ax1.bar(x - width/2, static, width, label='Static', color='#e74c3c', edgecolor='black')
    bars2 = ax1.bar(x + width/2, rl, width, label='RL Agent', color='#2ecc71', edgecolor='black')
    
    ax1.set_ylabel('Mean Latency (ms)', fontsize=12)
    ax1.set_title('Query Latency Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([w.title() for w in workloads], fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', fontsize=10)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', fontsize=10)
    
    # 2. Improvement percentage
    ax2 = fig.add_subplot(2, 2, 2)
    improvements = [(results[w]["static"]["mean"] - results[w]["rl"]["mean"]) / 
                   results[w]["static"]["mean"] * 100 for w in workloads]
    colors = ['#27ae60' if i > 10 else '#f39c12' if i > 0 else '#e74c3c' for i in improvements]
    bars = ax2.bar([w.title() for w in workloads], improvements, color=colors, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_ylabel('Latency Reduction (%)', fontsize=12)
    ax2.set_title('RL Agent Improvement Over Static', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(min(min(improvements) - 10, -5), max(max(improvements) + 10, 70))
    
    for bar, val in zip(bars, improvements):
        ypos = bar.get_height() + 2 if val >= 0 else bar.get_height() - 5
        ax2.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    # 3. Speedup factor
    ax3 = fig.add_subplot(2, 2, 3)
    speedups = [results[w]["static"]["mean"] / results[w]["rl"]["mean"] for w in workloads]
    colors = ['#27ae60' if s > 1.5 else '#f39c12' if s > 1 else '#e74c3c' for s in speedups]
    bars = ax3.bar([w.title() for w in workloads], speedups, color=colors, edgecolor='black', linewidth=1.5)
    ax3.axhline(y=1, color='black', linewidth=1, linestyle='--', label='No improvement')
    ax3.set_ylabel('Speedup Factor (×)', fontsize=12)
    ax3.set_title('RL Speedup Over Static', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    
    for bar, val in zip(bars, speedups):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}×', ha='center', fontsize=11, fontweight='bold')
    
    # 4. Summary statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calculate overall stats
    total_static = sum(results[w]["static"]["mean"] for w in workloads)
    total_rl = sum(results[w]["rl"]["mean"] for w in workloads)
    avg_improvement = (total_static - total_rl) / total_static * 100
    best_improvement = max(improvements)
    best_workload = workloads[improvements.index(best_improvement)]
    
    summary_text = f"""
    PERFORMANCE SUMMARY
    ═══════════════════════════════════════
    
    Average Latency Reduction:  {avg_improvement:.1f}%
    Best Improvement:           {best_improvement:.1f}% ({best_workload})
    
    Average Static Latency:     {total_static/4:.2f} ms
    Average RL Latency:         {total_rl/4:.2f} ms
    
    Workloads Improved:         {sum(1 for i in improvements if i > 0)}/4
    Workloads Unchanged:        {sum(1 for i in improvements if abs(i) < 1)}/4
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=13,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.suptitle('Morph: RL-Based Database Partition Optimization', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path / "advanced_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved results to {save_path / 'advanced_results.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-timesteps', type=int, default=200000)
    parser.add_argument('--output-dir', type=str, default='./advanced_training')
    parser.add_argument('--specialist', action='store_true', 
                       help='Train separate specialists for each workload')
    args = parser.parse_args()
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ppo_agent import PPOAgent, PPOConfig, train
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.output_dir) / f"run_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    if args.specialist:
        # Train specialists for each workload
        agents = {}
        for workload in ["skewed", "shifting", "bimodal"]:
            agents[workload] = train_specialist(workload, args.total_timesteps // 3, save_path)
        
        # Evaluate each specialist on its workload
        results = {}
        for workload in ["uniform", "skewed", "shifting", "bimodal"]:
            static = evaluate_agent(None, workload)
            
            # Use the appropriate specialist
            if workload in agents:
                rl = evaluate_agent(agents[workload], workload)
            else:
                rl = static  # No specialist for uniform
            
            results[workload] = {"static": static, "rl": rl}
    else:
        # Train single generalist on mixed workloads
        logger.info("="*60)
        logger.info("TRAINING GENERALIST AGENT")
        logger.info("="*60)
        
        # Create environment that samples different workloads
        class MixedEnv(AdvancedMorphEnv):
            def reset(self, seed=None, options=None):
                self.workload_type = np.random.choice(
                    ["skewed", "shifting", "bimodal"],
                    p=[0.5, 0.3, 0.2]
                )
                return super().reset(seed, options)
        
        env = MixedEnv(episode_length=100)
        
        config = PPOConfig(
            obs_dim=14,
            num_action_types=6,
            num_continuous_params=4,
            hidden_dims=[256, 256],
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=15,
            gamma=0.995,
            gae_lambda=0.98,
            entropy_coef=0.02,
        )
        
        agent = PPOAgent(config)
        agent = train(env, agent, args.total_timesteps, log_interval=50,
                      save_path=str(save_path / "generalist"))
        
        # Evaluate
        logger.info("\n" + "="*60)
        logger.info("EVALUATION")
        logger.info("="*60)
        
        results = {}
        for workload in ["uniform", "skewed", "shifting", "bimodal"]:
            static = evaluate_agent(None, workload)
            rl = evaluate_agent(agent, workload)
            results[workload] = {"static": static, "rl": rl}
            
            improvement = (static["mean"] - rl["mean"]) / static["mean"] * 100
            logger.info(f"{workload:12s}: Static={static['mean']:.2f}ms, "
                       f"RL={rl['mean']:.2f}ms ({improvement:+.1f}%)")
    
    # Plot and save
    plot_results(results, save_path)
    
    with open(save_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    total_static = 0
    total_rl = 0
    
    print(f"\n{'Workload':<12} {'Static':<12} {'RL':<12} {'Improvement':<12} {'Speedup':<10}")
    print("-"*58)
    
    for w in ["uniform", "skewed", "shifting", "bimodal"]:
        s = results[w]["static"]["mean"]
        r = results[w]["rl"]["mean"]
        imp = (s - r) / s * 100
        speedup = s / r
        total_static += s
        total_rl += r
        print(f"{w:<12} {s:<12.2f} {r:<12.2f} {imp:+.1f}%{'':<6} {speedup:.2f}×")
    
    avg_imp = (total_static - total_rl) / total_static * 100
    avg_speedup = total_static / total_rl
    print("-"*58)
    print(f"{'AVERAGE':<12} {total_static/4:<12.2f} {total_rl/4:<12.2f} {avg_imp:+.1f}%{'':<6} {avg_speedup:.2f}×")
    
    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
