"""
Quick test to verify training setup works correctly.
Tests random workload selection and TrainParameters.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env import PartitionEnv
from train import TrainParameters
import numpy as np


def test_random_workload_selection():
    """Test 1: Verify environment can randomly select workloads."""
    print("\n=== Test 1: Random Workload Selection ===")
    try:
        # Create environment without specifying workload
        env = PartitionEnv()
        print(f"✓ Environment created with random workload")
        print(f"  Workload type: {env.workload.__class__.__name__}")

        # Reset and check that workload changes
        workload_types = []
        for i in range(10):
            obs, info = env.reset()
            workload_types.append(env.workload.__class__.__name__)

        # Check we got different workload types
        unique_workloads = set(workload_types)
        print(f"✓ After 10 resets, got {len(unique_workloads)} different workload types:")
        for wl in unique_workloads:
            count = workload_types.count(wl)
            print(f"    {wl}: {count} times")

        if len(unique_workloads) > 1:
            print("✓ Random workload selection is working (got multiple types)")
        else:
            print("⚠ Only got one workload type (may be random chance)")

        env.close()
        return True
    except Exception as e:
        print(f"✗ Random workload test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_train_parameters():
    """Test 2: Verify TrainParameters dataclass works."""
    print("\n=== Test 2: TrainParameters Dataclass ===")
    try:
        # Create with defaults
        params = TrainParameters()
        print("✓ Created TrainParameters with defaults")
        print(f"  Total timesteps: {params.total_timesteps}")
        print(f"  Learning rate: {params.learning_rate}")
        print(f"  Checkpoint dir: {params.checkpoint_dir}")

        # Create with custom values
        custom_params = TrainParameters(
            total_timesteps=10,
            learning_rate=0.001,
            checkpoint_dir="./test_checkpoints"
        )
        print("✓ Created TrainParameters with custom values")
        print(f"  Total timesteps: {custom_params.total_timesteps}")
        print(f"  Learning rate: {custom_params.learning_rate}")
        print(f"  Checkpoint dir: {custom_params.checkpoint_dir}")

        return True
    except Exception as e:
        print(f"✗ TrainParameters test failed: {e}")
        return False


def test_env_with_specific_workload():
    """Test 3: Verify environment can still use specific workload."""
    print("\n=== Test 3: Environment with Specific Workload ===")
    try:
        from workloads import GaussianWorkload

        # Create workload with specific parameters
        workload = GaussianWorkload(mean=0.7, std=0.1)
        env = PartitionEnv(workload=workload)
        print("✓ Created environment with specific Gaussian workload")
        print(f"  Workload: {env.workload.__class__.__name__}")
        print(f"  Mean: {env.workload.mean:.2f}, Std: {env.workload.std:.2f}")

        # Reset without specifying workload (should select random)
        obs, info = env.reset()
        print(f"✓ Reset without workload - now using: {env.workload.__class__.__name__}")

        # Reset with specific workload
        specific_workload = GaussianWorkload(mean=0.3, std=0.05)
        obs, info = env.reset(workload=specific_workload)
        print(f"✓ Reset with specific workload - using: {env.workload.__class__.__name__}")
        print(f"  Mean: {env.workload.mean:.2f}, Std: {env.workload.std:.2f}")

        env.close()
        return True
    except Exception as e:
        print(f"✗ Specific workload test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training_loop():
    """Test 4: Run a mini training loop (2 episodes)."""
    print("\n=== Test 4: Mini Training Loop ===")
    try:
        from stable_baselines3 import PPO

        # Create environment
        env = PartitionEnv()
        print("✓ Created environment")

        # Create simple PPO model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0003,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            verbose=0,
            device="cpu",
        )
        print("✓ Created PPO model")

        # Train for just 2 steps
        print("  Training for 2 timesteps...")
        model.learn(total_timesteps=2, progress_bar=False)
        print("✓ Training completed successfully")

        # Test prediction
        obs, info = env.reset()
        action, _states = model.predict(obs)
        print(f"✓ Model can predict actions: {action}")

        env.close()
        return True
    except Exception as e:
        print(f"✗ Mini training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("TRAINING SETUP TEST SUITE")
    print("=" * 60)

    tests = [
        test_random_workload_selection,
        test_train_parameters,
        test_env_with_specific_workload,
        test_mini_training_loop,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        print("\nYou can now run full training with:")
        print("  python train.py")
        return True
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
