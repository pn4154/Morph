import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
import db_utils
from workloads import (
    GaussianWorkload,
    SlidingGaussianWorkload,
    BimodalWorkload,
    UniformWorkload,
    GaussianRangeWorkload,
    SlidingGaussianRangeWorkload,
    BimodalRangeWorkload,
    UniformRangeWorkload,
)


class PartitionEnv(gym.Env):
    """Custom Gym environment for learning optimal database partitioning.

    State: [boundary0, boundary1, p10, p25, p50, p75, p90]
    Action: [new_boundary1, new_boundary2] (continuous values in [0,1])
    Reward: Negative average query latency
    """

    metadata = {"render_modes": []}

    def __init__(self, workload=None, num_orders=10000, queries_per_step=500):
        """Initializes Gym environment with action/observation spaces.
        Sets up database connection and initial state.

        Args:
            workload: Optional workload object that implements sample(max_id) method.
                     If None, a random workload will be selected.
            num_orders: Maximum OrderID in database (default: 10000)
            queries_per_step: Number of queries to execute per step (default: 500)
        """
        super().__init__()

        # Store or randomly select workload
        if workload is None:
            self.workload = self.selectRandomWorkload()
        else:
            self.workload = workload
        self.max_id = num_orders
        self.queries_per_step = queries_per_step

        # Action space: 2 continuous values in [0, 1] for partition boundaries
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: 7 values (all normalized [0, 1])
        # [boundary0, boundary1, percentile_10, percentile_25, percentile_50,
        #  percentile_75, percentile_90]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # Database connection
        self.conn = None

        # Tracking for observations
        self.accessed_order_ids = []
        self.last_latencies = []

    def reset(self, seed=None, options=None, workload=None):
        """Resets database to initial state with default partitions.
        Returns initial observation (7 values).

        Args:
            seed: Random seed for environment
            options: Additional options
            workload: Optional workload to use for this episode.
                     If None, a random workload will be selected.
        """
        super().reset(seed=seed)

        # Select or use provided workload
        if workload is not None:
            self.workload = workload
        else:
            self.workload = self.selectRandomWorkload()

        # Close old connection if exists
        if self.conn:
            self.conn.close()

        # Get fresh connection
        self.conn = db_utils.get_connection()

        # Reset to equal partitions (33%, 66%)
        db_utils.repartition(self.conn, 0.333, 0.666)

        # Clear tracking
        self.accessed_order_ids = []
        self.last_latencies = []

        # Get initial observation
        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        """Applies action (2 floats), repartitions database.
        Runs QUERIES_PER_STEP queries, measures performance.
        Returns (obs, reward, done, truncated, info)."""

        # Extract and sort boundaries from action
        boundary1, boundary2 = float(action[0]), float(action[1])
        boundaries = sorted([boundary1, boundary2])
        b1, b2 = boundaries

        # Apply repartitioning
        db_utils.repartition(self.conn, b1, b2)

        # Run workload
        self._run_workload()

        # Calculate reward
        reward = self._calculate_reward()

        # Get new observation
        observation = self._get_observation()

        # Episode always terminates after one step (for now)
        terminated = True
        truncated = False

        info = {
            "avg_latency": np.mean(self.last_latencies),
            "boundaries": [b1, b2],
            "num_queries": len(self.last_latencies),
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Collects current state: boundaries and percentiles of access patterns.
        Returns numpy array of 7 normalized values [0, 1]."""

        # Get current partition boundaries
        boundaries = db_utils.get_partition_boundaries(self.conn)

        # Calculate percentiles of accessed OrderIDs
        if len(self.accessed_order_ids) > 0:
            # Calculate percentiles (10th, 25th, 50th, 75th, 90th)
            percentiles = np.percentile(
                self.accessed_order_ids, [10, 25, 50, 75, 90]
            )
            # Normalize to [0, 1]
            p10, p25, p50, p75, p90 = percentiles / self.max_id
        else:
            # Default values if no queries yet (uniform distribution assumption)
            p10, p25, p50, p75, p90 = 0.1, 0.25, 0.5, 0.75, 0.9

        # Construct observation: [boundary0, boundary1, p10, p25, p50, p75, p90]
        observation = np.array(
            [
                boundaries[0],  # boundary 0 (normalized)
                boundaries[1],  # boundary 1 (normalized)
                p10,           # 10th percentile (normalized)
                p25,           # 25th percentile (normalized)
                p50,           # 50th percentile (normalized)
                p75,           # 75th percentile (normalized)
                p90,           # 90th percentile (normalized)
            ],
            dtype=np.float32,
        )

        return observation

    def _calculate_reward(self):
        """Computes reward as negative average query latency.
        Lower latency = higher reward."""

        if len(self.last_latencies) == 0:
            return 0.0

        avg_latency_ms = np.mean(self.last_latencies)

        # Reward is negative latency (we want to minimize latency)
        reward = -avg_latency_ms

        return reward

    def selectRandomWorkload(self):
        """Randomly selects one of the eight workload types with random parameters.

        Returns:
            Workload object (Gaussian, SlidingGaussian, Bimodal, Uniform, or their Range variants)
        """
        workload_classes = [
            GaussianWorkload,
            SlidingGaussianWorkload,
            BimodalWorkload,
            UniformWorkload,
            GaussianRangeWorkload,
            SlidingGaussianRangeWorkload,
            BimodalRangeWorkload,
            UniformRangeWorkload,
        ]

        # Randomly select workload class
        WorkloadClass = random.choice(workload_classes)

        # Create instance with random parameters (handled by each workload's __init__)
        return WorkloadClass()

    def _run_workload(self):
        """Executes queries from the workload (self.queries_per_step queries).
        Records latencies and access patterns for observation."""

        self.accessed_order_ids = []
        self.last_latencies = []

        for _ in range(self.queries_per_step):
            # Get query from workload (returns SQL string with OrderID embedded)
            query = self.workload.sample(self.max_id)

            # Extract OrderID(s) from query for tracking
            if "BETWEEN" in query:
                # Range query format: "SELECT * FROM Orders WHERE OrderID BETWEEN X AND Y"
                # Extract lower and upper bounds
                between_part = query.split("BETWEEN")[1].split("AND")
                lower_bound = int(between_part[0].strip())
                upper_bound = int(between_part[1].strip())
                # Use midpoint for tracking
                order_id = (lower_bound + upper_bound) // 2
            else:
                # Point query format: "SELECT * FROM Orders WHERE OrderID = X"
                order_id = int(query.split("=")[1].strip())

            self.accessed_order_ids.append(order_id)

            # Execute query (query already has OrderID, so no params needed)
            _, latency_ms = db_utils.execute_query(self.conn, query, None)

            self.last_latencies.append(latency_ms)

    def close(self):
        """Closes database connection.
        Cleanup on environment termination."""
        if self.conn:
            self.conn.close()
            self.conn = None
