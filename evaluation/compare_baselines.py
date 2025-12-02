#!/usr/bin/env python3
"""
Morph Baseline Comparison

Compares the trained RL agent against:
1. Uniform (Round-Robin) partitioning - evenly distribute shards
2. Static partitioning - no changes, use default hash distribution
3. Random partitioning - random shard movements

Generates comparison plots and metrics.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineStrategy:
    """Base class for partitioning strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.episode_rewards = []
        self.latencies = []
        self.actions_taken = []
    
    def select_action(self, obs: np.ndarray, env) -> np.ndarray:
        raise NotImplementedError
    
    def reset(self):
        self.episode_rewards = []
        self.latencies = []
        self.actions_taken = []


class StaticStrategy(BaselineStrategy):
    """Static partitioning - always NO_OP (no changes)"""
    
    def __init__(self):
        super().__init__("Static (No Changes)")
    
    def select_action(self, obs: np.ndarray, env) -> np.ndarray:
        # Action 0 = NO_OP
        action = np.zeros(10, dtype=np.float32)
        action[0] = 1.0  # NO_OP has highest weight
        return action


class UniformStrategy(BaselineStrategy):
    """Uniform partitioning - always trigger rebalance to distribute evenly"""
    
    def __init__(self):
        super().__init__("Uniform (Rebalance)")
        self.rebalance_interval = 10  # Rebalance every N steps
        self.step_count = 0
    
    def select_action(self, obs: np.ndarray, env) -> np.ndarray:
        self.step_count += 1
        action = np.zeros(10, dtype=np.float32)
        
        if self.step_count % self.rebalance_interval == 0:
            # Action 4 = REBALANCE_PARTITION
            action[4] = 1.0
        else:
            # NO_OP
            action[0] = 1.0
        
        return action
    
    def reset(self):
        super().reset()
        self.step_count = 0


class RandomStrategy(BaselineStrategy):
    """Random partitioning - random actions"""
    
    def __init__(self):
        super().__init__("Random")
    
    def select_action(self, obs: np.ndarray, env) -> np.ndarray:
        return env.action_space.sample()


class RLAgentStrategy(BaselineStrategy):
    """Trained RL agent"""
    
    def __init__(self, agent):
        super().__init__("RL Agent (PPO)")
        self.agent = agent
    
    def select_action(self, obs: np.ndarray, env) -> np.ndarray:
        action, _, _ = self.agent.select_action(obs, deterministic=True)
        return action


def evaluate_strategy(
    env,
    strategy: BaselineStrategy,
    n_episodes: int = 10,
    max_steps: int = 100
) -> Dict[str, Any]:
    """Evaluate a strategy over multiple episodes"""
    
    strategy.reset()
    all_rewards = []
    all_latencies = []
    action_counts = {i: 0 for i in range(6)}
    episode_lengths = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_latencies = []
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            action = strategy.select_action(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # Track action type
            action_type = int(np.argmax(action[:6]))
            action_counts[action_type] += 1
            
            # Track latency if available
            if 'mean_latency_ms' in info:
                episode_latencies.append(info['mean_latency_ms'])
        
        all_rewards.append(episode_reward)
        all_latencies.extend(episode_latencies)
        episode_lengths.append(steps)
        
        logger.info(f"  {strategy.name} - Episode {ep+1}: Reward = {episode_reward:.2f}")
    
    return {
        "strategy": strategy.name,
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "min_reward": float(np.min(all_rewards)),
        "max_reward": float(np.max(all_rewards)),
        "mean_latency_ms": float(np.mean(all_latencies)) if all_latencies else 0,
        "p95_latency_ms": float(np.percentile(all_latencies, 95)) if all_latencies else 0,
        "action_distribution": action_counts,
        "episode_rewards": all_rewards,
        "mean_episode_length": float(np.mean(episode_lengths))
    }


def plot_comparison(results: List[Dict], save_path: Path):
    """Generate comparison plots"""
    
    strategies = [r["strategy"] for r in results]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    # ============ 1. Reward Comparison ============
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Bar chart of mean rewards
    ax = axes[0, 0]
    mean_rewards = [r["mean_reward"] for r in results]
    std_rewards = [r["std_reward"] for r in results]
    x = np.arange(len(strategies))
    bars = ax.bar(x, mean_rewards, yerr=std_rewards, capsize=5, color=colors[:len(strategies)])
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Mean Episode Reward (± std)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, mean_rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Box plot of rewards
    ax = axes[0, 1]
    reward_data = [r["episode_rewards"] for r in results]
    bp = ax.boxplot(reward_data, labels=strategies, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Episode Reward')
    ax.set_title('Reward Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Action distribution comparison
    ax = axes[1, 0]
    action_names = ['NO_OP', 'MOVE', 'SPLIT', 'MERGE', 'REBAL', 'RANGE']
    x = np.arange(len(action_names))
    width = 0.2
    
    for i, r in enumerate(results):
        counts = [r["action_distribution"].get(j, 0) for j in range(6)]
        total = sum(counts) if sum(counts) > 0 else 1
        percentages = [c/total * 100 for c in counts]
        ax.bar(x + i*width, percentages, width, label=r["strategy"], color=colors[i])
    
    ax.set_xticks(x + width * (len(results)-1) / 2)
    ax.set_xticklabels(action_names)
    ax.set_ylabel('Action Frequency (%)')
    ax.set_title('Action Distribution by Strategy')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Latency comparison (if available)
    ax = axes[1, 1]
    latencies = [r["mean_latency_ms"] for r in results]
    if any(l > 0 for l in latencies):
        bars = ax.bar(x[:len(strategies)], latencies, color=colors[:len(strategies)])
        ax.set_xticks(x[:len(strategies)])
        ax.set_xticklabels(strategies, rotation=15, ha='right')
        ax.set_ylabel('Mean Latency (ms)')
        ax.set_title('Query Latency by Strategy')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        # Show reward improvement instead
        baseline_reward = results[0]["mean_reward"]  # Static as baseline
        improvements = [(r["mean_reward"] - baseline_reward) / abs(baseline_reward) * 100 
                       if baseline_reward != 0 else 0 for r in results]
        bars = ax.bar(strategies, improvements, color=colors[:len(strategies)])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Improvement over Static (%)')
        ax.set_title('Relative Performance Improvement')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.suptitle('Partitioning Strategy Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / "baseline_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============ 2. Summary Table Plot ============
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Strategy', 'Mean Reward', 'Std', 'Min', 'Max', 'vs Static']
    
    static_reward = results[0]["mean_reward"]
    for r in results:
        improvement = ((r["mean_reward"] - static_reward) / abs(static_reward) * 100 
                      if static_reward != 0 else 0)
        table_data.append([
            r["strategy"],
            f'{r["mean_reward"]:.2f}',
            f'{r["std_reward"]:.2f}',
            f'{r["min_reward"]:.2f}',
            f'{r["max_reward"]:.2f}',
            f'{improvement:+.1f}%'
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * len(headers)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Highlight best performer
    best_idx = np.argmax([r["mean_reward"] for r in results])
    for j in range(len(headers)):
        table[(best_idx + 1, j)].set_facecolor('#d4edda')
    
    plt.title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path / "comparison_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison plots saved to {save_path}")


def create_simulated_env(num_workers=3, episode_length=100):
    """Create simulated environment for comparison"""
    import gymnasium as gym
    from gymnasium.spaces import Box
    
    class SimulatedMorphEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.num_workers = num_workers
            self.episode_length = episode_length
            
            obs_dim = num_workers * 2 + 8
            self.observation_space = Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
            self.action_space = Box(low=-1, high=1, shape=(10,), dtype=np.float32)
            
            self.step_count = 0
            self.shard_distribution = None
        
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.step_count = 0
            self.shard_distribution = np.random.dirichlet(np.ones(self.num_workers))
            return self._get_obs(), {}
        
        def _get_obs(self):
            return np.concatenate([
                self.shard_distribution,
                np.random.uniform(0.2, 0.8, self.num_workers),
                np.array([0.1, 0.1, 0.2, 0.3]),
                np.array([0.5, 0.5, 0.5, self.step_count / self.episode_length])
            ]).astype(np.float32)
        
        def step(self, action):
            self.step_count += 1
            action_type = int(np.argmax(action[:6]))
            params = (action[6:10] + 1) / 2
            
            # Simulate effects
            if action_type == 1:  # MOVE_PARTITION
                source = int(params[0] * self.num_workers) % self.num_workers
                target = int(params[1] * self.num_workers) % self.num_workers
                if source != target and self.shard_distribution[source] > 0.1:
                    self.shard_distribution[source] -= 0.1
                    self.shard_distribution[target] += 0.1
            elif action_type == 4:  # REBALANCE
                self.shard_distribution = np.ones(self.num_workers) / self.num_workers
            
            # Reward based on balance
            balance_score = 1.0 - np.std(self.shard_distribution) * 3
            reward = balance_score + np.random.randn() * 0.1
            
            done = self.step_count >= self.episode_length
            
            info = {
                "action_type": action_type,
                "mean_latency_ms": 50 + np.random.randn() * 10
            }
            
            return self._get_obs(), reward, False, done, info
    
    return SimulatedMorphEnv()


def main():
    parser = argparse.ArgumentParser(description='Compare RL agent with baselines')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained agent checkpoint')
    parser.add_argument('--n-episodes', type=int, default=20,
                       help='Number of episodes per strategy')
    parser.add_argument('--episode-length', type=int, default=100,
                       help='Max steps per episode')
    parser.add_argument('--output-dir', type=str, default='./comparison_results',
                       help='Output directory')
    parser.add_argument('--simulate', action='store_true',
                       help='Use simulated environment')
    parser.add_argument('--coordinator-host', type=str, default='localhost')
    parser.add_argument('--coordinator-port', type=int, default=5555)
    
    args = parser.parse_args()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.output_dir) / f"comparison_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load trained agent
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "rl_agent"))
    from ppo_agent import PPOAgent, PPOConfig
    
    logger.info(f"Loading agent from {args.checkpoint}")
    
    # Create config and agent
    config = PPOConfig(obs_dim=14, num_action_types=6, num_continuous_params=4)
    agent = PPOAgent(config)
    agent.load(args.checkpoint)
    
    # Create environment
    if args.simulate:
        logger.info("Using simulated environment")
        env = create_simulated_env(episode_length=args.episode_length)
    else:
        from citus_env import MorphEnvFlat
        env = MorphEnvFlat(
            coordinator_host=args.coordinator_host,
            coordinator_port=args.coordinator_port,
            episode_length=args.episode_length
        )
    
    # Define strategies to compare
    strategies = [
        StaticStrategy(),
        UniformStrategy(),
        RandomStrategy(),
        RLAgentStrategy(agent)
    ]
    
    # Evaluate each strategy
    results = []
    for strategy in strategies:
        logger.info(f"\nEvaluating: {strategy.name}")
        result = evaluate_strategy(env, strategy, n_episodes=args.n_episodes)
        results.append(result)
        logger.info(f"  Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
    
    # Save results
    with open(save_path / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    plot_comparison(results, save_path)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    static_reward = results[0]["mean_reward"]
    for r in results:
        improvement = ((r["mean_reward"] - static_reward) / abs(static_reward) * 100 
                      if static_reward != 0 else 0)
        print(f"\n{r['strategy']}:")
        print(f"  Mean Reward: {r['mean_reward']:.2f} ± {r['std_reward']:.2f}")
        print(f"  vs Static: {improvement:+.1f}%")
    
    print("\n" + "="*60)
    print(f"Results saved to: {save_path}")
    print(f"  - baseline_comparison.png")
    print(f"  - comparison_table.png")
    print(f"  - comparison_results.json")
    
    env.close()


if __name__ == "__main__":
    main()
