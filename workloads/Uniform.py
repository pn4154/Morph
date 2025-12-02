"""
Uniform workload: Samples OrderIDs uniformly across the entire range.
"""

import numpy as np
from .WorkloadAbstractClass import WorkloadAbstractClass


class UniformWorkload(WorkloadAbstractClass):
    """Workload that samples uniformly from the entire OrderID range.

    No parameters needed - all OrderIDs have equal probability.
    """

    def __init__(self):
        """Initialize Uniform workload."""
        super().__init__()

    def sample(self, max_id):
        """Sample OrderID from Uniform distribution.

        Args:
            max_id: Maximum OrderID in database

        Returns:
            SQL query string for point lookup
        """
        # Sample uniformly from [0, 1]
        normalized_value = np.random.uniform(0.0, 1.0)

        # Convert to OrderID
        order_id = self._normalize_to_id(normalized_value, max_id)

        # Build and return query
        return self._build_query(order_id)
