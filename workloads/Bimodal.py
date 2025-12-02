"""
Bimodal workload: Samples from two Gaussian distributions.
Each sample randomly selects one of the two modes.
"""

import numpy as np
from .WorkloadAbstractClass import WorkloadAbstractClass


class BimodalWorkload(WorkloadAbstractClass):
    """Workload that samples from a bimodal distribution (mixture of two Gaussians).

    Each sample randomly chooses between mode1 and mode2 with equal probability.

    Parameters:
        mean1: Center of first Gaussian (normalized [0, 1])
        std1: Standard deviation of first Gaussian (normalized [0, 1])
        mean2: Center of second Gaussian (normalized [0, 1])
        std2: Standard deviation of second Gaussian (normalized [0, 1])
    """

    # Default parameter ranges for random initialization
    MEAN1_MIN = 0.15
    MEAN1_MAX = 0.45
    MEAN2_MIN = 0.55
    MEAN2_MAX = 0.85
    STD_MIN = 0.05
    STD_MAX = 0.15

    def __init__(self, mean1=None, std1=None, mean2=None, std2=None):
        """Initialize Bimodal workload.

        Args:
            mean1: Mean of first mode [0, 1]. If None, randomly selected.
            std1: Std dev of first mode [0, 1]. If None, randomly selected.
            mean2: Mean of second mode [0, 1]. If None, randomly selected.
            std2: Std dev of second mode [0, 1]. If None, randomly selected.
        """
        super().__init__()

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
        """Sample OrderID from Bimodal distribution.

        Args:
            max_id: Maximum OrderID in database

        Returns:
            SQL query string for point lookup
        """
        # Randomly select which mode to sample from (50/50)
        if np.random.random() < 0.5:
            # Sample from first mode
            normalized_value = np.random.normal(self.mean1, self.std1)
        else:
            # Sample from second mode
            normalized_value = np.random.normal(self.mean2, self.std2)

        # Convert to OrderID
        order_id = self._normalize_to_id(normalized_value, max_id)

        # Build and return query
        return self._build_query(order_id)
