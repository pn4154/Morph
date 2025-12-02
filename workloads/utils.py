"""
Utility functions for workload sampling.
"""

import random


def select_random_range(value, max_id, max_range_size=0.01):
    """Select a random range query based on a normalized value.

    Args:
        value: Normalized value [0, 1] representing the center of the range
        max_id: Maximum OrderID in database
        max_range_size: Maximum size of range as fraction [0, 1] (default: 0.01 = 1%)

    Returns:
        SQL query string for range lookup

    Example:
        If max_range_size = 0.01:
        - Random range = 0.0075 (random value between 0 and 0.01)
        - lower_bound_max = 1 - 0.0075 = 0.9925
        - lower_bound = value * 0.9925
        - upper_bound = lower_bound + 0.0075
        - Query: SELECT * FROM Orders WHERE OrderID BETWEEN lower_bound AND upper_bound
    """
    # Generate random range size between 0 and max_range_size
    range_size = random.uniform(0, max_range_size)

    # Calculate maximum value for lower bound (to ensure upper bound doesn't exceed 1.0)
    lower_bound_max = 1.0 - range_size

    # Calculate lower bound (normalized [0, 1])
    lower_bound_normalized = min(value, lower_bound_max)

    # Calculate upper bound (normalized [0, 1])
    upper_bound_normalized = lower_bound_normalized + range_size

    # Convert to actual OrderIDs [1, max_id]
    lower_bound = max(1, int(lower_bound_normalized * max_id))
    upper_bound = min(max_id, int(upper_bound_normalized * max_id))

    # Ensure upper_bound > lower_bound
    if upper_bound <= lower_bound:
        upper_bound = lower_bound + 1

    # Build SQL query
    query = f"SELECT * FROM Orders WHERE OrderID BETWEEN {lower_bound} AND {upper_bound}"

    return query
