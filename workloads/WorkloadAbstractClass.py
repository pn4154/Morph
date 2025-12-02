"""
Abstract base class for workload distributions.
All workload classes must inherit from this and implement the sample method.
"""

from abc import ABC, abstractmethod


class WorkloadAbstractClass(ABC):
    """Base class for all workload types.

    Each workload samples from a distribution and generates SQL queries
    for point lookups on Orders table by OrderID.
    """

    def __init__(self):
        """Initialize workload. Subclasses should set their own parameters."""
        pass

    @abstractmethod
    def sample(self, max_id):
        """Sample from the workload distribution and return SQL query string.

        Args:
            max_id: Maximum OrderID in the database (e.g., 10000)

        Returns:
            SQL query string for point lookup, e.g.:
            "SELECT * FROM Orders WHERE OrderID = 5432"
        """
        pass

    def _normalize_to_id(self, normalized_value, max_id):
        """Convert normalized value [0, 1] to integer OrderID [1, max_id].

        Args:
            normalized_value: Float in range [0, 1]
            max_id: Maximum OrderID in database

        Returns:
            Integer OrderID in range [1, max_id]
        """
        # Clip to [0, 1] range
        normalized_value = max(0.0, min(1.0, normalized_value))

        # Convert to [1, max_id]
        order_id = int(normalized_value * (max_id - 1)) + 1

        return order_id

    def _build_query(self, order_id):
        """Build SQL query string for point lookup.

        Args:
            order_id: The OrderID to query

        Returns:
            SQL query string
        """
        return f"SELECT * FROM Orders WHERE OrderID = {order_id}"
