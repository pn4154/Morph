"""
Gaussian workload: Samples OrderIDs from a Gaussian (normal) distribution.
"""

import numpy as np
from .WorkloadAbstractClass import WorkloadAbstractClass


class GaussianWorkload(WorkloadAbstractClass):
    """Workload that samples from a Gaussian distribution.

    Parameters:
        mean: Center of the distribution (normalized [0, 1])
        std: Standard deviation (normalized [0, 1])
    """

    # Default parameter ranges for random initialization
    MEAN_MIN = 0.2
    MEAN_MAX = 0.8
    STD_MIN = 0.05
    STD_MAX = 0.25

    def __init__(self, mean=None, std=None):
        """Initialize Gaussian workload.

        Args:
            mean: Mean of distribution [0, 1]. If None, randomly selected.
            std: Standard deviation [0, 1]. If None, randomly selected.
        """
        super().__init__()

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
        """Sample OrderID from Gaussian distribution.

        Args:
            max_id: Maximum OrderID in database

        Returns:
            SQL query string for point lookup
        """
        # Sample from Gaussian distribution
        normalized_value = np.random.normal(self.mean, self.std)

        # Convert to OrderID
        order_id = self._normalize_to_id(normalized_value, max_id)

        # Build and return query
        return self._build_query(order_id)
