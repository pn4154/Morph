"""
Sliding Gaussian workload: Gaussian distribution with mean that slides over time.
The mean moves by 'speed' each query and bounces when hitting boundaries.
"""

import numpy as np
from .WorkloadAbstractClass import WorkloadAbstractClass


class SlidingGaussianWorkload(WorkloadAbstractClass):
    """Workload with a Gaussian distribution that slides over time.

    The mean starts at starting_mean and moves by speed each sample.
    When it reaches boundary [0 or 1], it reverses direction.

    Parameters:
        starting_mean: Initial center of distribution (normalized [0, 1])
        std: Standard deviation (normalized [0, 1])
        speed: Amount mean moves per query (normalized [0, 1])
    """

    # Default parameter ranges for random initialization
    STARTING_MEAN_MIN = 0.2
    STARTING_MEAN_MAX = 0.8
    STD_MIN = 0.05
    STD_MAX = 0.2
    SPEED_MIN = 0.001
    SPEED_MAX = 0.01

    def __init__(self, starting_mean=None, std=None, speed=None):
        """Initialize Sliding Gaussian workload.

        Args:
            starting_mean: Initial mean [0, 1]. If None, randomly selected.
            std: Standard deviation [0, 1]. If None, randomly selected.
            speed: Movement speed per query [0, 1]. If None, randomly selected.
        """
        super().__init__()

        # Set or randomly select parameters
        if starting_mean is None:
            self.starting_mean = np.random.uniform(
                self.STARTING_MEAN_MIN, self.STARTING_MEAN_MAX
            )
        else:
            self.starting_mean = starting_mean

        if std is None:
            self.std = np.random.uniform(self.STD_MIN, self.STD_MAX)
        else:
            self.std = std

        if speed is None:
            self.speed = np.random.uniform(self.SPEED_MIN, self.SPEED_MAX)
        else:
            self.speed = speed

        # Initialize current state
        self.current_mean = self.starting_mean
        self.direction = 1  # 1 for right, -1 for left

    def sample(self, max_id):
        """Sample OrderID from Sliding Gaussian distribution.

        Args:
            max_id: Maximum OrderID in database

        Returns:
            SQL query string for point lookup
        """
        # Sample from Gaussian at current mean
        normalized_value = np.random.normal(self.current_mean, self.std)

        # Convert to OrderID
        order_id = self._normalize_to_id(normalized_value, max_id)

        # Update mean for next sample
        self.current_mean += self.speed * self.direction

        # Bounce if hitting boundaries
        if self.current_mean >= 1.0:
            self.current_mean = 1.0
            self.direction = -1  # Reverse direction
        elif self.current_mean <= 0.0:
            self.current_mean = 0.0
            self.direction = 1  # Reverse direction

        # Build and return query
        return self._build_query(order_id)
