"""
Gaussian range workload: Samples range queries from a Gaussian (normal) distribution.
"""

import numpy as np
from .WorkloadAbstractClassRange import WorkloadAbstractClassRange
from .utils import select_random_range


class GaussianRangeWorkload(WorkloadAbstractClassRange):
    """Range workload that samples from a Gaussian distribution.

    Parameters:
        mean: Center of the distribution (normalized [0, 1])
        std: Standard deviation (normalized [0, 1])
        max_range_size: Maximum size of range as fraction [0, 1] (default: 0.01 = 1%)
    """

    # Default parameter ranges for random initialization
    MEAN_MIN = 0.2
    MEAN_MAX = 0.8
    STD_MIN = 0.05
    STD_MAX = 0.25

    def __init__(self, mean=None, std=None, max_range_size=0.01):
        """Initialize Gaussian range workload.

        Args:
            mean: Mean of distribution [0, 1]. If None, randomly selected.
            std: Standard deviation [0, 1]. If None, randomly selected.
            max_range_size: Maximum size of range as fraction [0, 1]
        """
        super().__init__(max_range_size=max_range_size)

        # Set or randomly select parameters
        if mean is None:
            self.mean = np.random.uniform(self.MEAN_MIN, self.MEAN_MAX)
        else:
            self.mean = mean

        if std is None:
            self.std = np.random.uniform(self.STD_MIN, self.STD_MAX)
        else:
            self.std = std

    def sample(self, max_id):
        """Sample range query from Gaussian distribution.

        Args:
            max_id: Maximum OrderID in database

        Returns:
            SQL query string for range lookup (BETWEEN query)
        """
        # Sample from Gaussian distribution (normalized [0, 1])
        normalized_value = np.random.normal(self.mean, self.std)

        # Normalize to [0, 1] range
        normalized_value = self._normalize_value(normalized_value)

        # Generate range query using utility function
        return select_random_range(normalized_value, max_id, self.max_range_size)
