"""
Test suite for database initialization and utilities.
Tests schema creation, partitioning, and query functionality.
"""

import sys
import os

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import db_utils
import init_db
import psycopg2


def test_database_creation():
    """Test 1: Verify database can be created and connected to."""
    print("\n=== Test 1: Database Creation ===")
    try:
        # Drop and create database
        init_db.drop_and_create_database()
        print(" Database created successfully")

        # Try to connect
        conn = db_utils.get_connection()
        print(" Connection established successfully")

        conn.close()
        return True
    except Exception as e:
        print(f" Database creation failed: {e}")
        return False


def test_table_creation():
    """Test 2: Verify all tables are created with correct schema."""
    print("\n=== Test 2: Table Creation ===")
    try:
        conn = db_utils.get_connection()
        cursor = conn.cursor()

        # Check if Customers table exists
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'customers'
            )
        """
        )
        customers_exists = cursor.fetchone()[0]
        if customers_exists:
            print(" Customers table exists")
        else:
            print(" Customers table missing")
            return False

        # Check if Products table exists
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'products'
            )
        """
        )
        products_exists = cursor.fetchone()[0]
        if products_exists:
            print(" Products table exists")
        else:
            print(" Products table missing")
            return False

        # Check if Orders table exists
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'orders'
            )
        """
        )
        orders_exists = cursor.fetchone()[0]
        if orders_exists:
            print(" Orders table exists")
        else:
            print(" Orders table missing")
            return False

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f" Table creation test failed: {e}")
        return False


def test_partitions():
    """Test 3: Verify Orders table has 3 partitions."""
    print("\n=== Test 3: Partition Creation ===")
    try:
        conn = db_utils.get_connection()
        cursor = conn.cursor()

        # Check for partition tables
        cursor.execute(
            """
            SELECT c.relname
            FROM pg_class c
            JOIN pg_inherits i ON c.oid = i.inhrelid
            JOIN pg_class parent ON i.inhparent = parent.oid
            WHERE parent.relname = 'orders'
            AND c.relname LIKE 'orders_p%'
            ORDER BY c.relname
        """
        )

        partitions = cursor.fetchall()
        partition_names = [p[0] for p in partitions]

        if len(partition_names) == 3:
            print(f" Found 3 partitions: {partition_names}")
        else:
            print(f" Expected 3 partitions, found {len(partition_names)}")
            return False

        # Verify partition boundaries
        boundaries = db_utils.get_partition_boundaries(conn)
        print(f" Partition boundaries: {boundaries}")

        if len(boundaries) == 2:
            print(" Correct number of boundaries (2)")
        else:
            print(f" Expected 2 boundaries, got {len(boundaries)}")
            return False

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f" Partition test failed: {e}")
        return False


def test_data_population():
    """Test 4: Verify data is populated in all tables."""
    print("\n=== Test 4: Data Population ===")
    try:
        conn = db_utils.get_connection()
        cursor = conn.cursor()

        # Check Customers count
        cursor.execute("SELECT COUNT(*) FROM Customers")
        customer_count = cursor.fetchone()[0]
        expected_customers = init_db.NUM_CUSTOMERS
        if customer_count == expected_customers:
            print(f" Customers: {customer_count}/{expected_customers}")
        else:
            print(f" Customers: {customer_count}/{expected_customers}")
            return False

        # Check Products count
        cursor.execute("SELECT COUNT(*) FROM Products")
        product_count = cursor.fetchone()[0]
        expected_products = init_db.NUM_PRODUCTS
        if product_count == expected_products:
            print(f" Products: {product_count}/{expected_products}")
        else:
            print(f" Products: {product_count}/{expected_products}")
            return False

        # Check Orders count
        cursor.execute("SELECT COUNT(*) FROM Orders")
        order_count = cursor.fetchone()[0]
        expected_orders = init_db.NUM_ORDERS
        if order_count == expected_orders:
            print(f" Orders: {order_count}/{expected_orders}")
        else:
            print(f" Orders: {order_count}/{expected_orders}")
            return False

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f" Data population test failed: {e}")
        return False


def test_query_execution():
    """Test 5: Verify queries can be executed and return results."""
    print("\n=== Test 5: Query Execution ===")
    try:
        conn = db_utils.get_connection()

        # Test a simple order lookup
        order_id = 1000
        query = "SELECT * FROM Orders WHERE OrderID = %s"
        results, latency = db_utils.execute_query(conn, query, (order_id,))

        if len(results) > 0:
            print(f" Query executed successfully (latency: {latency:.2f}ms)")
            print(f"  Result: OrderID={results[0][0]}, CustomerId={results[0][1]}")
        else:
            print(f" Query returned no results")
            return False

        # Test multiple queries
        latencies = []
        for i in range(1, 11):
            _, latency = db_utils.execute_query(conn, query, (i * 100,))
            latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies)
        print(f" 10 queries executed, avg latency: {avg_latency:.2f}ms")

        conn.close()
        return True
    except Exception as e:
        print(f" Query execution test failed: {e}")
        return False


def test_repartitioning():
    """Test 6: Verify repartitioning works correctly."""
    print("\n=== Test 6: Repartitioning ===")
    try:
        conn = db_utils.get_connection()

        # Get initial boundaries
        initial_boundaries = db_utils.get_partition_boundaries(conn)
        print(f"Initial boundaries: {initial_boundaries}")

        # Repartition with new boundaries
        new_b1, new_b2 = 0.25, 0.75
        print(f"\nRepartitioning to [{new_b1}, {new_b2}]...")
        db_utils.repartition(conn, new_b1, new_b2)

        # Verify new boundaries
        updated_boundaries = db_utils.get_partition_boundaries(conn)
        print(f"Updated boundaries: {updated_boundaries}")

        # Check if boundaries are approximately correct (within 0.01)
        if abs(updated_boundaries[0] - new_b1) < 0.01 and abs(
            updated_boundaries[1] - new_b2
        ) < 0.01:
            print(" Repartitioning successful")
        else:
            print(f" Boundaries don't match expected values")
            return False

        # Verify data is still accessible after repartitioning
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Orders")
        order_count = cursor.fetchone()[0]

        if order_count == init_db.NUM_ORDERS:
            print(f" All {order_count} orders still accessible after repartition")
        else:
            print(f" Data lost during repartitioning: {order_count}/{init_db.NUM_ORDERS}")
            return False

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f" Repartitioning test failed: {e}")
        return False


def test_partition_distribution():
    """Test 7: Verify orders are distributed across partitions."""
    print("\n=== Test 7: Partition Distribution ===")
    try:
        conn = db_utils.get_connection()
        cursor = conn.cursor()

        # Reset to equal partitions
        db_utils.repartition(conn, 0.333, 0.666)

        # Check row count in each partition
        cursor.execute("SELECT COUNT(*) FROM orders_p1")
        p1_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM orders_p2")
        p2_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM orders_p3")
        p3_count = cursor.fetchone()[0]

        total = p1_count + p2_count + p3_count

        print(f"Partition 1: {p1_count} orders ({p1_count/total*100:.1f}%)")
        print(f"Partition 2: {p2_count} orders ({p2_count/total*100:.1f}%)")
        print(f"Partition 3: {p3_count} orders ({p3_count/total*100:.1f}%)")

        if total == init_db.NUM_ORDERS:
            print(f" Total orders match: {total}")
        else:
            print(f" Total mismatch: {total}/{init_db.NUM_ORDERS}")
            return False

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f" Partition distribution test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("DATABASE INITIALIZATION TEST SUITE")
    print("=" * 60)

    # Test 1: Create database
    if not test_database_creation():
        print("\n Cannot proceed without database")
        return False

    # Initialize database
    print("\n" + "=" * 60)
    print("INITIALIZING DATABASE")
    print("=" * 60)
    conn = db_utils.get_connection()
    init_db.create_tables(conn)
    init_db.populate_tables(conn)
    conn.close()

    # Run remaining tests
    tests = [
        test_table_creation,
        test_partitions,
        test_data_population,
        test_query_execution,
        test_repartitioning,
        test_partition_distribution,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results) + 1  # +1 for database creation
    total = len(results) + 1
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n ALL TESTS PASSED!")
        return True
    else:
        print(f"\n {total - passed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
