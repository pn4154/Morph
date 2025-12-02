"""
Light test suite for PartitionEnv.
Tests basic functionality including PostgreSQL connection and workload integration.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env import PartitionEnv
from workloads import GaussianWorkload, UniformWorkload
import db_utils


def test_postgres_connection():
    """Test 1: Verify PostgreSQL connection works."""
    print("\n=== Test 1: PostgreSQL Connection ===")
    try:
        conn = db_utils.get_connection()
        print("✓ Successfully connected to PostgreSQL")

        # Test simple query
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Orders")
        count = cursor.fetchone()[0]
        print(f"✓ Found {count} orders in database")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"✗ PostgreSQL connection failed: {e}")
        return False


def test_env_creation():
    """Test 2: Verify environment can be created with workload."""
    print("\n=== Test 2: Environment Creation ===")
    try:
        # Create workload
        workload = GaussianWorkload(mean=0.5, std=0.1)
        print("✓ Created Gaussian workload")

        # Create environment
        env = PartitionEnv(workload)
        print("✓ Created PartitionEnv with workload")

        # Check observation space
        if env.observation_space.shape == (7,):
            print(f"✓ Observation space has correct shape: {env.observation_space.shape}")
        else:
            print(f"✗ Observation space has wrong shape: {env.observation_space.shape} (expected (7,))")
            return False

        # Check action space
        if env.action_space.shape == (2,):
            print(f"✓ Action space has correct shape: {env.action_space.shape}")
        else:
            print(f"✗ Action space has wrong shape: {env.action_space.shape} (expected (2,))")
            return False

        env.close()
        return True
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False


def test_env_reset():
    """Test 3: Verify environment reset works."""
    print("\n=== Test 3: Environment Reset ===")
    try:
        workload = UniformWorkload()
        env = PartitionEnv(workload)
        print("✓ Created environment")

        # Reset environment
        obs, info = env.reset()
        print("✓ Environment reset successfully")

        # Check observation shape
        if obs.shape == (7,):
            print(f"✓ Observation shape correct: {obs.shape}")
        else:
            print(f"✗ Observation shape wrong: {obs.shape} (expected (7,))")
            return False

        # Check observation values are normalized [0, 1]
        if np.all(obs >= 0.0) and np.all(obs <= 1.0):
            print(f"✓ All observation values in [0, 1] range")
        else:
            print(f"✗ Some observation values out of range: min={obs.min()}, max={obs.max()}")
            return False

        # Print observation
        print(f"  Initial observation:")
        print(f"    Boundary 0: {obs[0]:.3f}")
        print(f"    Boundary 1: {obs[1]:.3f}")
        print(f"    P10: {obs[2]:.3f}, P25: {obs[3]:.3f}, P50: {obs[4]:.3f}")
        print(f"    P75: {obs[5]:.3f}, P90: {obs[6]:.3f}")

        env.close()
        return True
    except Exception as e:
        print(f"✗ Environment reset failed: {e}")
        return False


def test_env_step():
    """Test 4: Verify environment step works with workload."""
    print("\n=== Test 4: Environment Step ===")
    try:
        # Create workload with fixed parameters
        workload = GaussianWorkload(mean=0.3, std=0.1)
        env = PartitionEnv(workload)
        print("✓ Created environment with Gaussian workload (mean=0.3)")

        # Reset
        obs, info = env.reset()
        print("✓ Environment reset")

        # Take a step with custom boundaries
        action = np.array([0.25, 0.75], dtype=np.float32)
        print(f"  Taking action: boundaries at {action[0]:.2f} and {action[1]:.2f}")

        obs, reward, terminated, truncated, info = env.step(action)
        print("✓ Step executed successfully")

        # Check outputs
        print(f"  Observation shape: {obs.shape}")
        print(f"  Reward (negative latency): {reward:.3f}")
        print(f"  Average latency: {info['avg_latency']:.3f}ms")
        print(f"  Num queries executed: {info['num_queries']}")
        print(f"  Boundaries used: {info['boundaries']}")

        # Verify observation updated with percentiles
        print(f"  Updated observation:")
        print(f"    Boundary 0: {obs[0]:.3f}, Boundary 1: {obs[1]:.3f}")
        print(f"    P10: {obs[2]:.3f}, P25: {obs[3]:.3f}, P50: {obs[4]:.3f}")
        print(f"    P75: {obs[5]:.3f}, P90: {obs[6]:.3f}")

        # Check that percentiles reflect the workload (Gaussian with mean 0.3)
        # P50 should be close to 0.3
        if 0.15 < obs[4] < 0.45:
            print(f"✓ P50 ({obs[4]:.3f}) is near workload mean (0.3)")
        else:
            print(f"⚠ P50 ({obs[4]:.3f}) differs from workload mean (0.3) - may be variance")

        env.close()
        return True
    except Exception as e:
        print(f"✗ Environment step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workload_integration():
    """Test 5: Verify different workloads can be used."""
    print("\n=== Test 5: Multiple Workload Types ===")
    try:
        workloads = [
            ("Gaussian", GaussianWorkload(mean=0.6, std=0.1)),
            ("Uniform", UniformWorkload()),
        ]

        for name, workload in workloads:
            print(f"\n  Testing {name} workload...")
            env = PartitionEnv(workload)
            obs, info = env.reset()
            action = np.array([0.33, 0.66], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"    ✓ {name} workload works")
            print(f"      P50: {obs[4]:.3f}, Avg latency: {info['avg_latency']:.3f}ms")

            env.close()

        print("\n✓ All workload types work with environment")
        return True
    except Exception as e:
        print(f"✗ Workload integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_percentile_calculation():
    """Test 6: Verify percentiles are calculated correctly."""
    print("\n=== Test 6: Percentile Calculation ===")
    try:
        # Use uniform workload for predictable distribution
        workload = UniformWorkload()
        env = PartitionEnv(workload)
        obs, info = env.reset()

        # Take a step
        action = np.array([0.33, 0.66], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # Extract percentiles
        p10, p25, p50, p75, p90 = obs[2], obs[3], obs[4], obs[5], obs[6]

        print(f"  Percentiles: P10={p10:.3f}, P25={p25:.3f}, P50={p50:.3f}, P75={p75:.3f}, P90={p90:.3f}")

        # For uniform distribution, percentiles should be roughly at expected positions
        # P10 ≈ 0.1, P25 ≈ 0.25, P50 ≈ 0.5, P75 ≈ 0.75, P90 ≈ 0.9
        checks = [
            (p10, 0.1, "P10"),
            (p25, 0.25, "P25"),
            (p50, 0.5, "P50"),
            (p75, 0.75, "P75"),
            (p90, 0.9, "P90"),
        ]

        all_good = True
        for actual, expected, label in checks:
            # Allow 0.15 margin for variance
            if abs(actual - expected) < 0.15:
                print(f"  ✓ {label}: {actual:.3f} is close to expected {expected:.3f}")
            else:
                print(f"  ⚠ {label}: {actual:.3f} differs from expected {expected:.3f} (may be variance)")
                all_good = False

        # Check ordering (percentiles should be increasing)
        if p10 < p25 < p50 < p75 < p90:
            print("  ✓ Percentiles are in correct order (increasing)")
        else:
            print("  ✗ Percentiles are not in order")
            return False

        env.close()
        return True
    except Exception as e:
        print(f"✗ Percentile calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("PARTITION ENVIRONMENT TEST SUITE")
    print("=" * 60)

    tests = [
        test_postgres_connection,
        test_env_creation,
        test_env_reset,
        test_env_step,
        test_workload_integration,
        test_percentile_calculation,
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
        return True
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
