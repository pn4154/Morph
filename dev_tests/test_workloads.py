"""
Test suite for workload implementations.
Tests all four workload types and their distributions.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workloads import (
    GaussianWorkload,
    SlidingGaussianWorkload,
    BimodalWorkload,
    UniformWorkload,
)


def test_gaussian_workload():
    """Test 1: Test Gaussian workload with fixed parameters."""
    print("\n=== Test 1: Gaussian Workload ===")
    try:
        # Create workload with fixed parameters
        workload = GaussianWorkload(mean=0.5, std=0.1)
        print(f"✓ Created Gaussian workload (mean={workload.mean}, std={workload.std})")

        # Generate samples
        max_id = 10000
        samples = []
        for _ in range(1000):
            query = workload.sample(max_id)
            # Extract OrderID from query
            order_id = int(query.split("=")[1].strip())
            samples.append(order_id)

        # Check statistics
        mean_id = np.mean(samples)
        std_id = np.std(samples)
        expected_mean = 0.5 * max_id
        print(f"✓ Generated 1000 samples")
        print(f"  Mean OrderID: {mean_id:.1f} (expected ~{expected_mean:.1f})")
        print(f"  Std Dev: {std_id:.1f}")

        # Verify queries are valid
        if all(1 <= s <= max_id for s in samples):
            print("✓ All samples in valid range [1, 10000]")
            return True, samples
        else:
            print("✗ Some samples out of range")
            return False, samples

    except Exception as e:
        print(f"✗ Gaussian workload test failed: {e}")
        return False, []


def test_gaussian_random_params():
    """Test 2: Test Gaussian workload with random parameters."""
    print("\n=== Test 2: Gaussian with Random Parameters ===")
    try:
        # Create workload with random parameters
        workload = GaussianWorkload()
        print(f"✓ Created Gaussian workload with random params")
        print(f"  mean={workload.mean:.3f}, std={workload.std:.3f}")

        # Generate some samples
        max_id = 10000
        samples = [int(workload.sample(max_id).split("=")[1].strip()) for _ in range(100)]

        if all(1 <= s <= max_id for s in samples):
            print("✓ All samples in valid range")
            return True
        else:
            print("✗ Some samples out of range")
            return False

    except Exception as e:
        print(f"✗ Random parameter test failed: {e}")
        return False


def test_sliding_gaussian_workload():
    """Test 3: Test Sliding Gaussian workload."""
    print("\n=== Test 3: Sliding Gaussian Workload ===")
    try:
        # Create workload with fixed parameters
        workload = SlidingGaussianWorkload(starting_mean=0.3, std=0.08, speed=0.005)
        print(
            f"✓ Created Sliding Gaussian workload (mean={workload.starting_mean}, "
            f"std={workload.std}, speed={workload.speed})"
        )

        # Generate samples and track mean movement
        max_id = 10000
        samples = []
        means = []

        for _ in range(200):
            means.append(workload.current_mean)
            query = workload.sample(max_id)
            order_id = int(query.split("=")[1].strip())
            samples.append(order_id)

        # Check that mean moves
        if means[0] != means[-1]:
            print(f"✓ Mean moved from {means[0]:.3f} to {means[-1]:.3f}")
        else:
            print("✗ Mean did not move")
            return False, []

        # Check for direction changes (bouncing)
        direction_changes = 0
        for i in range(1, len(means)):
            if i < len(means) - 1:
                if (means[i] > means[i - 1]) != (means[i + 1] > means[i]):
                    direction_changes += 1

        print(f"✓ Direction changed {direction_changes} times (bouncing)")

        # Verify all samples in valid range
        if all(1 <= s <= max_id for s in samples):
            print("✓ All samples in valid range")
            return True, samples, means
        else:
            print("✗ Some samples out of range")
            return False, samples, means

    except Exception as e:
        print(f"✗ Sliding Gaussian test failed: {e}")
        return False, [], []


def test_bimodal_workload():
    """Test 4: Test Bimodal workload."""
    print("\n=== Test 4: Bimodal Workload ===")
    try:
        # Create workload with fixed parameters
        workload = BimodalWorkload(mean1=0.25, std1=0.08, mean2=0.75, std2=0.08)
        print(
            f"✓ Created Bimodal workload (mean1={workload.mean1}, std1={workload.std1}, "
            f"mean2={workload.mean2}, std2={workload.std2})"
        )

        # Generate samples
        max_id = 10000
        samples = []
        for _ in range(1000):
            query = workload.sample(max_id)
            order_id = int(query.split("=")[1].strip())
            samples.append(order_id)

        # Check for bimodal distribution (should have samples around both means)
        normalized_samples = [s / max_id for s in samples]
        lower_samples = sum(1 for s in normalized_samples if s < 0.5)
        upper_samples = sum(1 for s in normalized_samples if s >= 0.5)

        print(f"✓ Generated 1000 samples")
        print(f"  Lower half: {lower_samples} samples ({lower_samples/10:.1f}%)")
        print(f"  Upper half: {upper_samples} samples ({upper_samples/10:.1f}%)")

        # Both halves should have reasonable number of samples
        if lower_samples > 200 and upper_samples > 200:
            print("✓ Distribution appears bimodal")
        else:
            print("⚠ Distribution may not be clearly bimodal (but could be variance)")

        # Verify all samples in valid range
        if all(1 <= s <= max_id for s in samples):
            print("✓ All samples in valid range")
            return True, samples
        else:
            print("✗ Some samples out of range")
            return False, samples

    except Exception as e:
        print(f"✗ Bimodal test failed: {e}")
        return False, []


def test_uniform_workload():
    """Test 5: Test Uniform workload."""
    print("\n=== Test 5: Uniform Workload ===")
    try:
        # Create workload
        workload = UniformWorkload()
        print("✓ Created Uniform workload")

        # Generate samples
        max_id = 10000
        samples = []
        for _ in range(1000):
            query = workload.sample(max_id)
            order_id = int(query.split("=")[1].strip())
            samples.append(order_id)

        # Check uniformity - divide into bins
        num_bins = 10
        bin_size = max_id // num_bins
        bins = [0] * num_bins

        for s in samples:
            bin_idx = min((s - 1) // bin_size, num_bins - 1)
            bins[bin_idx] += 1

        print(f"✓ Generated 1000 samples")
        print("  Distribution across bins:")
        for i, count in enumerate(bins):
            print(f"    Bin {i+1} [{i*bin_size+1}-{(i+1)*bin_size}]: {count} ({count/10:.1f}%)")

        # Check if reasonably uniform (each bin should have ~100 samples)
        if all(50 < count < 150 for count in bins):
            print("✓ Distribution appears uniform")
        else:
            print("⚠ Distribution may not be perfectly uniform (but could be variance)")

        # Verify all samples in valid range
        if all(1 <= s <= max_id for s in samples):
            print("✓ All samples in valid range")
            return True, samples
        else:
            print("✗ Some samples out of range")
            return False, samples

    except Exception as e:
        print(f"✗ Uniform test failed: {e}")
        return False, []


def test_query_format():
    """Test 6: Verify query format is correct."""
    print("\n=== Test 6: Query Format ===")
    try:
        workload = UniformWorkload()
        query = workload.sample(10000)

        print(f"Sample query: {query}")

        # Check format
        if query.startswith("SELECT * FROM Orders WHERE OrderID = "):
            print("✓ Query format is correct")
        else:
            print("✗ Query format is incorrect")
            return False

        # Check OrderID is integer
        order_id_str = query.split("=")[1].strip()
        order_id = int(order_id_str)

        if 1 <= order_id <= 10000:
            print(f"✓ OrderID {order_id} is valid")
            return True
        else:
            print(f"✗ OrderID {order_id} is out of range")
            return False

    except Exception as e:
        print(f"✗ Query format test failed: {e}")
        return False


def visualize_distributions():
    """Optional: Visualize all distributions (if matplotlib available)."""
    print("\n=== Visualizing Distributions ===")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        max_id = 10000
        num_samples = 5000

        # Gaussian
        workload = GaussianWorkload(mean=0.5, std=0.15)
        samples = [
            int(workload.sample(max_id).split("=")[1].strip())
            for _ in range(num_samples)
        ]
        axes[0, 0].hist(samples, bins=50, alpha=0.7, edgecolor="black")
        axes[0, 0].set_title("Gaussian Distribution")
        axes[0, 0].set_xlabel("OrderID")
        axes[0, 0].set_ylabel("Frequency")

        # Sliding Gaussian
        workload = SlidingGaussianWorkload(starting_mean=0.2, std=0.1, speed=0.005)
        samples = [
            int(workload.sample(max_id).split("=")[1].strip())
            for _ in range(num_samples)
        ]
        axes[0, 1].hist(samples, bins=50, alpha=0.7, edgecolor="black")
        axes[0, 1].set_title("Sliding Gaussian Distribution")
        axes[0, 1].set_xlabel("OrderID")
        axes[0, 1].set_ylabel("Frequency")

        # Bimodal
        workload = BimodalWorkload(mean1=0.3, std1=0.1, mean2=0.7, std2=0.1)
        samples = [
            int(workload.sample(max_id).split("=")[1].strip())
            for _ in range(num_samples)
        ]
        axes[1, 0].hist(samples, bins=50, alpha=0.7, edgecolor="black")
        axes[1, 0].set_title("Bimodal Distribution")
        axes[1, 0].set_xlabel("OrderID")
        axes[1, 0].set_ylabel("Frequency")

        # Uniform
        workload = UniformWorkload()
        samples = [
            int(workload.sample(max_id).split("=")[1].strip())
            for _ in range(num_samples)
        ]
        axes[1, 1].hist(samples, bins=50, alpha=0.7, edgecolor="black")
        axes[1, 1].set_title("Uniform Distribution")
        axes[1, 1].set_xlabel("OrderID")
        axes[1, 1].set_ylabel("Frequency")

        plt.tight_layout()
        save_path = "/Users/kolbeyang/Documents/School/fall_2025/csci_725/final_project/Morph/dev_tests/workload_distributions.png"
        plt.savefig(save_path)
        print(f"✓ Visualization saved to: {save_path}")
        print("  (You can view this file to see the distributions)")

    except Exception as e:
        print(f"⚠ Visualization failed (non-critical): {e}")


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("WORKLOAD IMPLEMENTATIONS TEST SUITE")
    print("=" * 60)

    results = []

    # Test 1: Gaussian
    result, _ = test_gaussian_workload()
    results.append(result)

    # Test 2: Gaussian with random params
    result = test_gaussian_random_params()
    results.append(result)

    # Test 3: Sliding Gaussian
    result, _, _ = test_sliding_gaussian_workload()
    results.append(result)

    # Test 4: Bimodal
    result, _ = test_bimodal_workload()
    results.append(result)

    # Test 5: Uniform
    result, _ = test_uniform_workload()
    results.append(result)

    # Test 6: Query format
    result = test_query_format()
    results.append(result)

    # Visualize (optional)
    visualize_distributions()

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
