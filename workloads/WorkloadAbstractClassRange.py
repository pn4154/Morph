"""
Abstract base class for range query workload distributions.
All range workload classes must inherit from this and implement the sample method.
"""

from abc import ABC, abstractmethod
from .utils import select_random_range


class WorkloadAbstractClassRange(ABC):
    """Base class for all range query workload types.

    Each workload samples from a distribution and generates SQL queries
    for range lookups on Orders table by OrderID.
    """

    def __init__(self, max_range_size=0.01):
        """Initialize workload with max range size.

        Args:
            max_range_size: Maximum size of range as fraction [0, 1] (default: 0.01 = 1%)
        """
        self.max_range_size = max_range_size

    @abstractmethod
    def sample(self, max_id):
        """Sample from the workload distribution and return SQL range query string.

        Args:
            max_id: Maximum OrderID in the database (e.g., 10000)

        Returns:
            SQL query string for range lookup, e.g.:
            "SELECT * FROM Orders WHERE OrderID BETWEEN 5000 AND 5100"
        """
        pass

    def _normalize_value(self, value):
        """Clip value to [0, 1] range.

        Args:
            value: Float value (possibly outside [0, 1])

        Returns:
            Float in range [0, 1]
        """
        return max(0.0, min(1.0, value))
