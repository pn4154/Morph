"""
Uniform range workload: Samples range queries uniformly across the entire range.
"""

import numpy as np
from .WorkloadAbstractClassRange import WorkloadAbstractClassRange
from .utils import select_random_range


class UniformRangeWorkload(WorkloadAbstractClassRange):
    """Range workload that samples uniformly from the entire OrderID range.

    Parameters:
        max_range_size: Maximum size of range as fraction [0, 1] (default: 0.01 = 1%)
    """

    def __init__(self, max_range_size=0.01):
        """Initialize Uniform range workload.

        Args:
            max_range_size: Maximum size of range as fraction [0, 1]
        """
        super().__init__(max_range_size=max_range_size)

    def sample(self, max_id):
        """Sample range query from Uniform distribution.

        Args:
            max_id: Maximum OrderID in database

        Returns:
            SQL query string for range lookup (BETWEEN query)
        """
        # Sample uniformly from [0, 1]
        normalized_value = np.random.uniform(0.0, 1.0)

        # Generate range query using utility function
        return select_random_range(normalized_value, max_id, self.max_range_size)
