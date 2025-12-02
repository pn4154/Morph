import psycopg2
import time
import numpy as np

# Constants
NUM_ORDERS = 10000
DB_NAME = "morph_db"
DB_USER = "postgres"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = 5432


def get_connection():
    """Returns psycopg2 connection to morph_db.
    Handles connection errors gracefully."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        raise


def execute_query(conn, query, params=None):
    """Executes a single query and returns results.
    Measures and returns execution time in milliseconds."""
    cursor = conn.cursor()

    start_time = time.time()
    cursor.execute(query, params)
    results = cursor.fetchall()
    end_time = time.time()

    latency_ms = (end_time - start_time) * 1000

    cursor.close()
    return results, latency_ms


def get_partition_boundaries(conn):
    """Queries pg_catalog to get current partition boundary values.
    Returns list of 2 floats representing boundaries normalized to [0,1]."""
    cursor = conn.cursor()

    # Get partition definitions from pg_class
    cursor.execute(
        """
        SELECT
            c.relname,
            pg_get_expr(c.relpartbound, c.oid) as partition_bound
        FROM pg_class c
        JOIN pg_inherits i ON c.oid = i.inhrelid
        JOIN pg_class parent ON i.inhparent = parent.oid
        WHERE parent.relname = 'orders'
        AND c.relname LIKE 'orders_p%'
        ORDER BY c.relname
    """
    )

    rows = cursor.fetchall()
    cursor.close()

    # Parse boundaries from partition definitions
    # Format: "FOR VALUES FROM (X) TO (Y)"
    boundaries = []
    for name, bound_expr in rows:
        if "TO" in bound_expr:
            # Extract the TO value (upper bound)
            to_part = bound_expr.split("TO")[1].strip()
            # Remove parentheses and convert to int
            upper_bound = int(to_part.strip("()' "))
            boundaries.append(upper_bound)

    # We have 3 partitions, so 2 boundaries (exclude the final upper limit)
    # Normalize to [0, 1]
    normalized = [b / NUM_ORDERS for b in boundaries[:2]]

    return normalized


def repartition(conn, boundary1, boundary2):
    """Drops old partitions and creates new ones with specified boundaries.
    Boundaries are floats [0,1], converts to actual order_id values."""
    cursor = conn.cursor()

    # Sort boundaries to ensure b1 < b2
    boundaries = sorted([boundary1, boundary2])
    b1_normalized, b2_normalized = boundaries

    # Convert to actual order_id values
    b1 = int(b1_normalized * NUM_ORDERS)
    b2 = int(b2_normalized * NUM_ORDERS)

    # Ensure boundaries are valid
    b1 = max(1, min(b1, NUM_ORDERS - 1))
    b2 = max(b1 + 1, min(b2, NUM_ORDERS))

    print(
        f"Repartitioning: boundaries at {b1} ({b1_normalized:.3f}), {b2} ({b2_normalized:.3f})"
    )

    # Create temporary table to hold all data
    cursor.execute("CREATE TEMP TABLE orders_temp AS SELECT * FROM orders")

    # Drop existing partitions (this deletes partition tables but temp has data)
    cursor.execute("DROP TABLE IF EXISTS orders_p1 CASCADE")
    cursor.execute("DROP TABLE IF EXISTS orders_p2 CASCADE")
    cursor.execute("DROP TABLE IF EXISTS orders_p3 CASCADE")

    # Create new partitions with updated boundaries
    cursor.execute(
        f"""
        CREATE TABLE orders_p1 PARTITION OF orders
        FOR VALUES FROM (0) TO ({b1})
    """
    )

    cursor.execute(
        f"""
        CREATE TABLE orders_p2 PARTITION OF orders
        FOR VALUES FROM ({b1}) TO ({b2})
    """
    )

    cursor.execute(
        f"""
        CREATE TABLE orders_p3 PARTITION OF orders
        FOR VALUES FROM ({b2}) TO ({NUM_ORDERS + 1})
    """
    )

    # Re-insert data from temp table (Postgres routes to correct partitions)
    cursor.execute("INSERT INTO orders SELECT * FROM orders_temp")

    # Drop temp table
    cursor.execute("DROP TABLE orders_temp")

    conn.commit()
    cursor.close()

    # print(f"Repartitioned: [0-{b1}), [{b1}-{b2}), [{b2}-{NUM_ORDERS + 1})")


def generate_skewed_order_id():
    """Generates random order_id with skew toward lower values.
    Uses beta distribution to create 80/20 skew."""
    # Beta distribution with alpha=2, beta=5 creates right skew (favors low values)
    # Scale to [1, NUM_ORDERS]
    normalized = np.random.beta(2, 5)
    order_id = int(normalized * (NUM_ORDERS - 1)) + 1
    return order_id
