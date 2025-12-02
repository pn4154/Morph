#!/usr/bin/env python3
"""
Morph Training Script

Main entry point for training the PPO agent on the Citus partition
optimization environment.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from citus_env import MorphEnvFlat, make_env
from ppo_agent import PPOAgent, PPOConfig, train, evaluate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Morph PPO Agent')
    
    # Environment settings
    parser.add_argument('--coordinator-host', type=str, default='localhost',
                        help='Citus coordinator hostname')
    parser.add_argument('--coordinator-port', type=int, default=5432,
                        help='Citus coordinator port')
    parser.add_argument('--database', type=str, default='morphdb',
                        help='Database name')
    parser.add_argument('--user', type=str, default='morph',
                        help='Database user')
    parser.add_argument('--password', type=str, default='MorphSecurePass123!',
                        help='Database password')
    parser.add_argument('--num-workers', type=int, default=3,
                        help='Number of Citus worker nodes')
    parser.add_argument('--workload-type', type=str, default='mixed',
                        choices=['oltp', 'olap', 'mixed'],
                        help='Workload type for training')
    parser.add_argument('--episode-length', type=int, default=100,
                        help='Maximum steps per episode')
    
    # Training settings
    parser.add_argument('--total-timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--n-steps', type=int, default=2048,
                        help='Steps per rollout')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Mini-batch size')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='PPO epochs per update')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                        help='PPO clip epsilon')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--value-coef', type=float, default=0.5,
                        help='Value loss coefficient')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cpu/cuda/auto)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Logging interval (episodes)')
    parser.add_argument('--save-interval', type=int, default=10000,
                        help='Checkpoint save interval (steps)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation')
    parser.add_argument('--simulate', action='store_true',
                        help='Use simulated environment (no actual DB)')
    
    return parser.parse_args()


def setup_device(device_str: str) -> str:
    """Setup compute device"""
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            logger.info("CUDA not available, using CPU")
    else:
        device = device_str
    
    return device


def create_simulated_env(args):
    """Create a simulated environment for testing without DB"""
    import gymnasium as gym
    from gymnasium.spaces import Box
    
    class SimulatedMorphEnv(gym.Env):
        """Simulated Morph environment for testing"""
        
        def __init__(self, num_workers=3, episode_length=100):
            super().__init__()
            self.num_workers = num_workers
            self.episode_length = episode_length
            
            obs_dim = num_workers * 2 + 8
            self.observation_space = Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
            self.action_space = Box(low=-1, high=1, shape=(10,), dtype=np.float32)
            
            self.step_count = 0
            self.shard_distribution = np.ones(num_workers) / num_workers
            self.baseline_latency = 100.0
        
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.step_count = 0
            self.shard_distribution = np.random.dirichlet(np.ones(self.num_workers))
            self.baseline_latency = 100.0 + np.random.randn() * 10
            return self._get_obs(), {"baseline_latency": self.baseline_latency}
        
        def _get_obs(self):
            obs = np.concatenate([
                self.shard_distribution,
                np.random.uniform(0.2, 0.8, self.num_workers),  # Node sizes
                np.array([0.1, 0.1, 0.2, 0.3]),  # Latency stats
                np.array([0.5, 0.5, 0.5, self.step_count / self.episode_length])
            ])
            return obs.astype(np.float32)
        
        def step(self, action):
            self.step_count += 1
            
            # Decode action
            action_type = int(np.argmax(action[:6]))
            params = (action[6:10] + 1) / 2
            
            # Simulate action effects
            if action_type == 1:  # MOVE_PARTITION
                # Simulate shard movement
                source = int(params[0] * self.num_workers) % self.num_workers
                target = int(params[1] * self.num_workers) % self.num_workers
                if source != target and self.shard_distribution[source] > 0.1:
                    amount = 0.1
                    self.shard_distribution[source] -= amount
                    self.shard_distribution[target] += amount
            elif action_type == 4:  # REBALANCE
                self.shard_distribution = np.ones(self.num_workers) / self.num_workers
            
            # Calculate reward based on balance
            balance_score = 1.0 - np.std(self.shard_distribution) * 3
            latency_score = -0.1 * np.random.uniform(0.5, 1.5)
            reward = balance_score + latency_score + np.random.randn() * 0.1
            
            # Check termination
            done = self.step_count >= self.episode_length
            
            info = {
                "action_type": ["NO_OP", "MOVE_PARTITION", "SPLIT_PARTITION", 
                               "MERGE_PARTITIONS", "REBALANCE_PARTITION", "MOVE_DATA_RANGE"][action_type],
                "action_success": True,
                "mean_latency_ms": 50 + np.random.randn() * 10,
                "shard_distribution": {f"worker-{i}": int(d * 100) 
                                       for i, d in enumerate(self.shard_distribution)}
            }
            
            return self._get_obs(), reward, False, done, info
    
    return SimulatedMorphEnv(
        num_workers=args.num_workers,
        episode_length=args.episode_length
    )


def plot_all_results(agent, save_path, eval_results=None):
    """Generate all training and evaluation plots"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # ============ 1. Training Curves ============
    if agent.training_history:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        history = agent.training_history
        
        axes[0, 0].plot([h['policy_loss'] for h in history], color='blue')
        axes[0, 0].set_title('Policy Loss')
        axes[0, 0].set_xlabel('Update')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot([h['value_loss'] for h in history], color='orange')
        axes[0, 1].set_title('Value Loss')
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot([h['entropy'] for h in history], color='green')
        axes[1, 0].set_title('Policy Entropy')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].grid(True, alpha=0.3)
        
        if 'approx_kl' in history[0]:
            axes[1, 1].plot([h['approx_kl'] for h in history], color='red')
            axes[1, 1].set_title('Approx KL Divergence')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('KL')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Training Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path / "training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved to {save_path / 'training_curves.png'}")
    
    # ============ 2. Episode Rewards ============
    if agent.episode_rewards:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        rewards = list(agent.episode_rewards)
        episodes = range(1, len(rewards) + 1)
        
        # Raw rewards with moving average
        axes[0].plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
        
        # Moving average
        window = min(10, len(rewards) // 5) if len(rewards) > 10 else 1
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window, len(rewards) + 1), moving_avg, 
                        color='red', linewidth=2, label=f'Moving Avg ({window})')
        
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Episode Rewards During Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Reward distribution histogram
        axes[1].hist(rewards, bins=30, color='blue', alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(rewards), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
        axes[1].axvline(np.median(rewards), color='green', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(rewards):.2f}')
        axes[1].set_xlabel('Reward')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Reward Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Training Episode Rewards', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path / "episode_rewards.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Episode rewards saved to {save_path / 'episode_rewards.png'}")
    
    # ============ 3. Evaluation Results ============
    if eval_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Action distribution pie chart
        action_dist = eval_results.get('action_distribution', {})
        if action_dist:
            action_names = ['NO_OP', 'MOVE_PARTITION', 'SPLIT_PARTITION', 
                          'MERGE_PARTITIONS', 'REBALANCE', 'MOVE_DATA_RANGE']
            counts = [action_dist.get(i, 0) for i in range(6)]
            
            # Filter out zero counts for pie chart
            non_zero = [(name, count) for name, count in zip(action_names, counts) if count > 0]
            if non_zero:
                names, values = zip(*non_zero)
                colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
                axes[0].pie(values, labels=names, autopct='%1.1f%%', colors=colors, startangle=90)
                axes[0].set_title('Action Distribution During Evaluation')
            else:
                axes[0].text(0.5, 0.5, 'No actions recorded', ha='center', va='center')
                axes[0].set_title('Action Distribution')
        
        # Evaluation metrics bar chart
        metrics = {
            'Mean Reward': eval_results.get('mean_reward', 0),
            'Std Reward': eval_results.get('std_reward', 0),
            'Mean Length': eval_results.get('mean_length', 0) / 10  # Scale down for display
        }
        
        bars = axes[1].bar(metrics.keys(), metrics.values(), color=['blue', 'orange', 'green'])
        axes[1].set_ylabel('Value')
        axes[1].set_title('Evaluation Metrics')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, metrics.values()):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Evaluation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path / "evaluation_results.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Evaluation results saved to {save_path / 'evaluation_results.png'}")
    
    # ============ 4. Combined Summary Plot ============
    fig = plt.figure(figsize=(16, 10))
    
    # Training reward curve
    ax1 = fig.add_subplot(2, 2, 1)
    if agent.episode_rewards:
        rewards = list(agent.episode_rewards)
        ax1.plot(rewards, alpha=0.3, color='blue')
        window = min(10, len(rewards) // 5) if len(rewards) > 10 else 1
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(rewards)), moving_avg, color='red', linewidth=2)
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
    
    # Policy loss
    ax2 = fig.add_subplot(2, 2, 2)
    if agent.training_history:
        ax2.plot([h['policy_loss'] for h in agent.training_history], color='blue')
        ax2.set_title('Policy Loss')
        ax2.set_xlabel('Update')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
    
    # Value loss
    ax3 = fig.add_subplot(2, 2, 3)
    if agent.training_history:
        ax3.plot([h['value_loss'] for h in agent.training_history], color='orange')
        ax3.set_title('Value Loss')
        ax3.set_xlabel('Update')
        ax3.set_ylabel('Loss')
        ax3.grid(True, alpha=0.3)
    
    # Action distribution
    ax4 = fig.add_subplot(2, 2, 4)
    if eval_results and eval_results.get('action_distribution'):
        action_names = ['NO_OP', 'MOVE', 'SPLIT', 'MERGE', 'REBAL', 'RANGE']
        counts = [eval_results['action_distribution'].get(i, 0) for i in range(6)]
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
        ax4.bar(action_names, counts, color=colors)
        ax4.set_title('Actions Taken (Evaluation)')
        ax4.set_xlabel('Action Type')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Morph RL Training Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / "summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Summary plot saved to {save_path / 'summary.png'}")


def main():
    args = parse_args()
    
    # Setup
    device = setup_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.save_dir) / f"morph_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(save_path / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"Save directory: {save_path}")
    
    # Create environment
    if args.simulate:
        logger.info("Using simulated environment")
        env = create_simulated_env(args)
    else:
        logger.info(f"Connecting to Citus cluster at {args.coordinator_host}:{args.coordinator_port}")
        try:
            env = MorphEnvFlat(
                coordinator_host=args.coordinator_host,
                coordinator_port=args.coordinator_port,
                database=args.database,
                user=args.user,
                password=args.password,
                num_workers=args.num_workers,
                episode_length=args.episode_length,
                workload_type=args.workload_type
            )
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            logger.info("Falling back to simulated environment")
            env = create_simulated_env(args)
    
    # Create agent config
    obs_dim = args.num_workers * 2 + 8
    config = PPOConfig(
        obs_dim=obs_dim,
        num_action_types=6,
        num_continuous_params=4,
        hidden_dims=[256, 256],
        activation="tanh",
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        total_timesteps=args.total_timesteps,
        device=device,
        seed=args.seed
    )
    
    # Create agent
    agent = PPOAgent(config)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        logger.info(f"Loading checkpoint from {args.load_checkpoint}")
        agent.load(args.load_checkpoint)
    
    if args.eval_only:
        # Evaluation only
        logger.info("Running evaluation only")
        results = evaluate(env, agent, n_episodes=args.eval_episodes)
        
        with open(save_path / "eval_results.json", "w") as f:
            results_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                          for k, v in results.items()}
            json.dump(results_json, f, indent=2)
        
        # Plot evaluation results
        plot_all_results(agent, save_path, results)
    else:
        # Training
        logger.info(f"Starting training for {args.total_timesteps} timesteps")
        
        agent = train(
            env=env,
            agent=agent,
            total_timesteps=args.total_timesteps,
            log_interval=args.log_interval,
            save_path=str(save_path),
            save_interval=args.save_interval
        )
        
        # Final evaluation
        logger.info("\nRunning final evaluation...")
        results = evaluate(env, agent, n_episodes=args.eval_episodes)
        
        with open(save_path / "final_eval_results.json", "w") as f:
            results_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                          for k, v in results.items()}
            json.dump(results_json, f, indent=2)
        
        # Generate all plots
        plot_all_results(agent, save_path, results)
        
        # Save episode rewards to JSON for later analysis
        with open(save_path / "episode_rewards.json", "w") as f:
            json.dump(list(agent.episode_rewards), f)
    
    # Cleanup
    env.close()
    
    logger.info(f"\nTraining complete! Results saved to {save_path}")
    logger.info(f"Generated plots:")
    logger.info(f"  - {save_path}/training_curves.png")
    logger.info(f"  - {save_path}/episode_rewards.png")
    logger.info(f"  - {save_path}/evaluation_results.png")
    logger.info(f"  - {save_path}/summary.png")


if __name__ == "__main__":
    main()
