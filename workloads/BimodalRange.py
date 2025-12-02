"""
Bimodal range workload: Samples range queries from two Gaussian distributions.
Each sample randomly selects one of the two modes.
"""

import numpy as np
from .WorkloadAbstractClassRange import WorkloadAbstractClassRange
from .utils import select_random_range


class BimodalRangeWorkload(WorkloadAbstractClassRange):
    """Range workload that samples from a bimodal distribution (mixture of two Gaussians).

    Each sample randomly chooses between mode1 and mode2 with equal probability.

    Parameters:
        mean1: Center of first Gaussian (normalized [0, 1])
        std1: Standard deviation of first Gaussian (normalized [0, 1])
        mean2: Center of second Gaussian (normalized [0, 1])
        std2: Standard deviation of second Gaussian (normalized [0, 1])
        max_range_size: Maximum size of range as fraction [0, 1] (default: 0.01 = 1%)
    """

    # Default parameter ranges for random initialization
    MEAN1_MIN = 0.15
    MEAN1_MAX = 0.45
    MEAN2_MIN = 0.55
    MEAN2_MAX = 0.85
    STD_MIN = 0.05
    STD_MAX = 0.15

    def __init__(self, mean1=None, std1=None, mean2=None, std2=None, max_range_size=0.01):
        """Initialize Bimodal range workload.

        Args:
            mean1: Mean of first mode [0, 1]. If None, randomly selected.
            std1: Std dev of first mode [0, 1]. If None, randomly selected.
            mean2: Mean of second mode [0, 1]. If None, randomly selected.
            std2: Std dev of second mode [0, 1]. If None, randomly selected.
            max_range_size: Maximum size of range as fraction [0, 1]
        """
        super().__init__(max_range_size=max_range_size)

        # Set or randomly select parameters for first mode
        if mean1 is None:
            self.mean1 = np.random.uniform(self.MEAN1_MIN, self.MEAN1_MAX)
        else:
            self.mean1 = mean1

        if std1 is None:
            self.std1 = np.random.uniform(self.STD_MIN, self.STD_MAX)
        else:
            self.std1 = std1

        # Set or randomly select parameters for second mode
        if mean2 is None:
            self.mean2 = np.random.uniform(self.MEAN2_MIN, self.MEAN2_MAX)
        else:
            self.mean2 = mean2

        if std2 is None:
            self.std2 = np.random.uniform(self.STD_MIN, self.STD_MAX)
        else:
            self.std2 = std2

    def sample(self, max_id):
        """Sample range query from Bimodal distribution.

        Args:
            max_id: Maximum OrderID in database

        Returns:
            SQL query string for range lookup (BETWEEN query)
        """
        # Randomly select which mode to sample from (50/50)
        if np.random.random() < 0.5:
            # Sample from first mode
            normalized_value = np.random.normal(self.mean1, self.std1)
        else:
            # Sample from second mode
            normalized_value = np.random.normal(self.mean2, self.std2)

        # Normalize to [0, 1] range
        normalized_value = self._normalize_value(normalized_value)

        # Generate range query using utility function
        return select_random_range(normalized_value, max_id, self.max_range_size)
