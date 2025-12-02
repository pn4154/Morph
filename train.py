"""
PPO training for adaptive database partitioning.
Trains agent to select optimal partition boundaries based on workload patterns.
"""

from dataclasses import dataclass
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from env import PartitionEnv
import os


@dataclass
class TrainParameters:
    """Hyperparameters for PPO training."""

    # Training parameters
    total_timesteps: int = 50  # Total episodes (50 episodes in CLAUDE.md)
    n_steps: int = 10  # Number of timesteps to collect before update
    batch_size: int = 10  # Batch size for training
    n_epochs: int = 3  # Number of optimization epochs per update

    # Environment parameters
    num_orders: int = 10000  # Maximum OrderID in database
    queries_per_step: int = 500  # Number of queries to execute per step

    # PPO hyperparameters
    learning_rate: float = 0.0003
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda parameter
    clip_range: float = 0.2  # PPO clipping parameter
    ent_coef: float = 0.0  # Entropy coefficient

    # Evaluation
    eval_freq: int = 10  # Evaluate every N episodes
    n_eval_episodes: int = 5  # Number of episodes for evaluation

    # Logging and saving
    tensorboard_log: str = "./ppo_partition_tensorboard/"
    checkpoint_dir: str = "./checkpoints"
    verbose: int = 1
    device: str = "cpu"


def train(params: TrainParameters):
    """Train PPO agent on database partitioning task.

    Environment randomly selects workloads (Gaussian, SlidingGaussian,
    Bimodal, Uniform) for each episode to learn robust partitioning.

    Args:
        params: TrainParameters object with hyperparameters
    """
    print("=" * 60)
    print("PPO TRAINING: ADAPTIVE DATABASE PARTITIONING")
    print("=" * 60)
    print(f"\nTraining Configuration:")
    print(f"  Total episodes: {params.total_timesteps}")
    print(f"  Learning rate: {params.learning_rate}")
    print(f"  N-steps: {params.n_steps}")
    print(f"  Batch size: {params.batch_size}")
    print(f"  Epochs per update: {params.n_epochs}")
    print(f"  Device: {params.device}")

    # Create checkpoint directory if it doesn't exist
    os.makedirs(params.checkpoint_dir, exist_ok=True)
    os.makedirs(params.tensorboard_log, exist_ok=True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nRun timestamp: {timestamp}")

    # Create training environment (will randomly select workloads)
    print("\n=== Creating Environment ===")
    print("Environment will randomly select workloads per episode:")
    print("  - GaussianWorkload")
    print("  - SlidingGaussianWorkload")
    print("  - BimodalWorkload")
    print("  - UniformWorkload")
    print(f"Environment config: num_orders={params.num_orders}, queries_per_step={params.queries_per_step}")
    env = PartitionEnv(
        num_orders=params.num_orders,
        queries_per_step=params.queries_per_step
    )
    print("✓ Training environment created")

    # Create evaluation environment (also uses random workloads)
    eval_env = PartitionEnv(
        num_orders=params.num_orders,
        queries_per_step=params.queries_per_step
    )
    print("✓ Evaluation environment created")

    # Create PPO model
    print("\n=== Initializing PPO Model ===")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=params.learning_rate,
        n_steps=params.n_steps,
        batch_size=params.batch_size,
        n_epochs=params.n_epochs,
        gamma=params.gamma,
        gae_lambda=params.gae_lambda,
        clip_range=params.clip_range,
        ent_coef=params.ent_coef,
        verbose=params.verbose,
        tensorboard_log=params.tensorboard_log,
        device=params.device,
    )
    print("✓ PPO model initialized")

    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(params.checkpoint_dir, f"best_{timestamp}"),
        log_path=os.path.join(params.checkpoint_dir, f"eval_{timestamp}"),
        eval_freq=params.eval_freq,
        n_eval_episodes=params.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    print("✓ Evaluation callback configured")

    # Train the model
    print("\n=== Starting Training ===")
    print(f"Training for {params.total_timesteps} episodes...")
    print(f"Progress will be logged to: {params.tensorboard_log}")
    print(f"To monitor: tensorboard --logdir {params.tensorboard_log}\n")

    model.learn(
        total_timesteps=params.total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    # Save final model with timestamp
    final_model_path = os.path.join(params.checkpoint_dir, f"model_{timestamp}")
    print(f"\n=== Saving Final Model ===")
    print(f"Saving to: {final_model_path}")
    model.save(final_model_path)
    print("✓ Model saved")

    # Cleanup
    env.close()
    eval_env.close()

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model saved to: {final_model_path}.zip")
    print(
        f"Best model saved to: {os.path.join(params.checkpoint_dir, f'best_{timestamp}')}"
    )
    print(f"TensorBoard logs: {params.tensorboard_log}")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)


if __name__ == "__main__":
    # Create parameters with sensible defaults
    params = TrainParameters(
        total_timesteps=300,  # 50 episodes as specified in CLAUDE.md
        learning_rate=0.0001,
        n_steps=10,
        batch_size=10,
        n_epochs=5,
        eval_freq=10,
        n_eval_episodes=5,
        verbose=1,
        device="cpu",
    )

    # Run training
    train(params)
