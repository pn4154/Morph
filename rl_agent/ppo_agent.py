"""
Morph PPO Agent: Proximal Policy Optimization for Database Partitioning

This module implements a PPO agent with a hybrid action space
(discrete action types + continuous parameters) for dynamic
partition management in Citus clusters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import logging
from collections import deque
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RolloutBuffer:
    """Buffer for storing rollout experiences"""
    
    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate tensors
        self.observations = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=device)
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add a transition to the buffer"""
        self.observations[self.ptr] = torch.from_numpy(obs).to(self.device)
        self.actions[self.ptr] = torch.from_numpy(action).to(self.device)
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """Compute GAE advantages and returns"""
        last_gae = 0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        
        self.returns[:self.size] = self.advantages[:self.size] + self.values[:self.size]
        
        # Normalize advantages
        adv_mean = self.advantages[:self.size].mean()
        adv_std = self.advantages[:self.size].std() + 1e-8
        self.advantages[:self.size] = (self.advantages[:self.size] - adv_mean) / adv_std
    
    def get_batches(self, batch_size: int):
        """Generate random mini-batches"""
        indices = np.random.permutation(self.size)
        
        for start in range(0, self.size, batch_size):
            end = min(start + batch_size, self.size)
            batch_indices = indices[start:end]
            
            yield (
                self.observations[batch_indices],
                self.actions[batch_indices],
                self.log_probs[batch_indices],
                self.advantages[batch_indices],
                self.returns[batch_indices],
                self.values[batch_indices]
            )
    
    def reset(self):
        """Reset the buffer"""
        self.ptr = 0
        self.size = 0


class ActorCriticNetwork(nn.Module):
    """
    Neural network for the PPO actor-critic with hybrid action space.
    
    Architecture:
    - Shared feature extractor (MLP)
    - Actor head (discrete action type + continuous parameters)
    - Critic head (state value)
    """
    
    def __init__(
        self,
        obs_dim: int,
        num_action_types: int = 6,
        num_continuous_params: int = 4,
        hidden_dims: List[int] = [256, 256],
        activation: str = "tanh"
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.num_action_types = num_action_types
        self.num_continuous_params = num_continuous_params
        
        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()
        
        # Shared feature extractor
        layers = []
        prev_dim = obs_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                self.activation
            ])
            prev_dim = dim
        self.shared = nn.Sequential(*layers)
        
        # Actor head - discrete action type
        self.action_type_head = nn.Linear(prev_dim, num_action_types)
        
        # Actor head - continuous parameters (mean and log_std)
        self.param_mean = nn.Linear(prev_dim, num_continuous_params)
        self.param_log_std = nn.Parameter(torch.zeros(num_continuous_params))
        
        # Critic head
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            self.activation,
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Smaller init for output heads
        nn.init.orthogonal_(self.action_type_head.weight, gain=0.01)
        nn.init.orthogonal_(self.param_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning:
        - action_type_logits: logits for discrete action type
        - param_mean: mean for continuous parameters
        - param_std: std for continuous parameters
        - value: state value estimate
        """
        features = self.shared(obs)
        
        action_type_logits = self.action_type_head(features)
        param_mean = torch.tanh(self.param_mean(features))  # Bound to [-1, 1]
        param_std = torch.exp(torch.clamp(self.param_log_std, -5, 2))
        value = self.value_head(features)
        
        return action_type_logits, param_mean, param_std, value.squeeze(-1)
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability and value.
        
        Returns:
        - action: combined action tensor [action_type_onehot(6), params(4)]
        - log_prob: log probability of the action
        - entropy: entropy of the policy
        - value: state value estimate
        """
        action_type_logits, param_mean, param_std, value = self.forward(obs)
        
        # Discrete action type distribution
        action_type_dist = Categorical(logits=action_type_logits)
        
        # Continuous parameter distribution
        param_dist = Normal(param_mean, param_std.expand_as(param_mean))
        
        if action is None:
            # Sample new action
            if deterministic:
                action_type = action_type_logits.argmax(dim=-1)
                params = param_mean
            else:
                action_type = action_type_dist.sample()
                params = param_dist.sample()
                params = torch.clamp(params, -1, 1)
            
            # Combine into single action tensor
            action_type_onehot = F.one_hot(action_type, self.num_action_types).float()
            action = torch.cat([action_type_onehot, params], dim=-1)
        else:
            # Extract action components from provided action
            action_type_onehot = action[:, :self.num_action_types]
            action_type = action_type_onehot.argmax(dim=-1)
            params = action[:, self.num_action_types:]
        
        # Compute log probabilities
        log_prob_type = action_type_dist.log_prob(action_type)
        log_prob_params = param_dist.log_prob(params).sum(dim=-1)
        log_prob = log_prob_type + log_prob_params
        
        # Compute entropy
        entropy_type = action_type_dist.entropy()
        entropy_params = param_dist.entropy().sum(dim=-1)
        entropy = entropy_type + entropy_params
        
        return action, log_prob, entropy, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get state value estimate only"""
        features = self.shared(obs)
        return self.value_head(features).squeeze(-1)


@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    # Environment
    obs_dim: int = 14  # num_workers * 2 + 8
    num_action_types: int = 6
    num_continuous_params: int = 4
    
    # Network
    hidden_dims: List[int] = None
    activation: str = "tanh"
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    total_timesteps: int = 1_000_000
    
    # Misc
    device: str = "cpu"
    seed: int = 42
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]


class PPOAgent:
    """
    PPO Agent for Morph database partition optimization.
    """
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Create network
        self.network = ActorCriticNetwork(
            obs_dim=config.obs_dim,
            num_action_types=config.num_action_types,
            num_continuous_params=config.num_continuous_params,
            hidden_dims=config.hidden_dims,
            activation=config.activation
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )
        
        # Rollout buffer
        action_dim = config.num_action_types + config.num_continuous_params
        self.buffer = RolloutBuffer(
            buffer_size=config.n_steps,
            obs_dim=config.obs_dim,
            action_dim=action_dim,
            device=self.device
        )
        
        # Training state
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.training_history = []
    
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action given observation.
        
        Returns:
        - action: numpy array of shape (10,)
        - log_prob: log probability of action
        - value: state value estimate
        """
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            action, log_prob, _, value = self.network.get_action_and_value(
                obs_tensor, deterministic=deterministic
            )
        
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item()
        )
    
    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout data.
        
        Returns dict of training metrics.
        """
        # Get last value for GAE computation
        # (In practice, this should use the final observation)
        last_obs = self.buffer.observations[self.buffer.ptr - 1]
        with torch.no_grad():
            last_value = self.network.get_value(last_obs.unsqueeze(0)).item()
        
        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(
            last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fractions = []
        
        # PPO update epochs
        for epoch in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                obs, actions, old_log_probs, advantages, returns, old_values = batch
                
                # Get current policy outputs
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    obs, action=actions
                )
                
                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon
                )
                policy_loss = -torch.min(
                    ratio * advantages,
                    clipped_ratio * advantages
                ).mean()
                
                # Value loss with clipping
                value_loss_unclipped = (new_values - returns) ** 2
                value_clipped = old_values + torch.clamp(
                    new_values - old_values,
                    -self.config.clip_epsilon,
                    self.config.clip_epsilon
                )
                value_loss_clipped = (value_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(
                    value_loss_unclipped,
                    value_loss_clipped
                ).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                # Track metrics
                with torch.no_grad():
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(-entropy_loss.item())
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    approx_kls.append(approx_kl)
                    clip_fractions.append(
                        ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item()
                    )
        
        # Reset buffer for next rollout
        self.buffer.reset()
        
        metrics = {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses),
            "approx_kl": np.mean(approx_kls),
            "clip_fraction": np.mean(clip_fractions)
        }
        
        self.training_history.append(metrics)
        
        return metrics
    
    def save(self, path: str):
        """Save agent state"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "training_history": self.training_history,
            "config": self.config.__dict__
        }, path / "agent.pt")
        
        # Save config separately as JSON for easy inspection
        with open(path / "config.json", "w") as f:
            config_dict = {k: v for k, v in self.config.__dict__.items() 
                          if not k.startswith("_")}
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent state"""
        path = Path(path)
        checkpoint = torch.load(path / "agent.pt", map_location=self.device, weights_only=False)
        
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        self.training_history = checkpoint["training_history"]
        
        logger.info(f"Agent loaded from {path}")


def train(
    env,
    agent: PPOAgent,
    total_timesteps: int,
    log_interval: int = 10,
    save_path: Optional[str] = None,
    save_interval: int = 50000
) -> PPOAgent:
    """
    Train the PPO agent on the environment.
    """
    config = agent.config
    
    obs, info = env.reset()
    episode_reward = 0
    episode_length = 0
    num_episodes = 0
    
    logger.info(f"Starting training for {total_timesteps} timesteps")
    
    while agent.total_steps < total_timesteps:
        # Collect rollout
        for step in range(config.n_steps):
            action, log_prob, value = agent.select_action(obs)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.buffer.add(obs, action, reward, value, log_prob, done)
            agent.total_steps += 1
            episode_reward += reward
            episode_length += 1
            
            if done:
                agent.episode_rewards.append(episode_reward)
                num_episodes += 1
                
                if num_episodes % log_interval == 0:
                    mean_reward = np.mean(list(agent.episode_rewards))
                    logger.info(
                        f"Episode {num_episodes} | "
                        f"Steps: {agent.total_steps} | "
                        f"Mean Reward: {mean_reward:.4f} | "
                        f"Episode Reward: {episode_reward:.4f}"
                    )
                
                obs, info = env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
        
        # Perform PPO update
        metrics = agent.update()
        
        logger.debug(
            f"Update | "
            f"Policy Loss: {metrics['policy_loss']:.4f} | "
            f"Value Loss: {metrics['value_loss']:.4f} | "
            f"Entropy: {metrics['entropy']:.4f}"
        )
        
        # Save checkpoint
        if save_path and agent.total_steps % save_interval < config.n_steps:
            agent.save(save_path)
    
    # Final save
    if save_path:
        agent.save(save_path)
    
    logger.info(f"Training complete. Total steps: {agent.total_steps}")
    
    return agent


def evaluate(
    env,
    agent: PPOAgent,
    n_episodes: int = 10,
    render: bool = False
) -> Dict[str, float]:
    """
    Evaluate the trained agent.
    """
    episode_rewards = []
    episode_lengths = []
    action_distributions = {i: 0 for i in range(6)}
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            
            # Track action type distribution
            action_type = int(np.argmax(action[:6]))
            action_distributions[action_type] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        logger.info(f"Episode {ep + 1}: Reward = {episode_reward:.4f}, Length = {episode_length}")
    
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "action_distribution": action_distributions
    }
    
    logger.info(f"\nEvaluation Results:")
    logger.info(f"  Mean Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    logger.info(f"  Mean Episode Length: {results['mean_length']:.1f}")
    logger.info(f"  Action Distribution: {action_distributions}")
    
    return results


if __name__ == "__main__":
    # Test the agent with a dummy environment
    from gymnasium.spaces import Box
    import gymnasium as gym
    
    # Create a simple test environment
    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = Box(low=0, high=1, shape=(14,), dtype=np.float32)
            self.action_space = Box(low=-1, high=1, shape=(10,), dtype=np.float32)
            self.step_count = 0
        
        def reset(self, seed=None, options=None):
            self.step_count = 0
            return np.random.rand(14).astype(np.float32), {}
        
        def step(self, action):
            self.step_count += 1
            reward = -np.sum(action**2) + np.random.randn() * 0.1
            done = self.step_count >= 100
            return np.random.rand(14).astype(np.float32), reward, done, False, {}
    
    env = DummyEnv()
    
    config = PPOConfig(
        obs_dim=14,
        n_steps=256,
        batch_size=32,
        n_epochs=4,
        total_timesteps=5000
    )
    
    agent = PPOAgent(config)
    
    # Quick training test
    agent = train(env, agent, total_timesteps=2000, log_interval=5)
    
    # Evaluate
    results = evaluate(env, agent, n_episodes=5)
    
    print("\nTest completed successfully!")
