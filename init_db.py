import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import random
from decimal import Decimal

# Constants
NUM_CUSTOMERS = 1000
NUM_PRODUCTS = 100
NUM_ORDERS = 10000
DB_NAME = "morph_db"
DB_USER = "postgres"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = 5432


def drop_and_create_database():
    """Drops existing database if exists and creates fresh one.
    Connects to default 'postgres' db to perform operation."""
    conn = psycopg2.connect(
        dbname="postgres",
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    # Drop if exists
    cursor.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
    print(f"Dropped database {DB_NAME} (if existed)")

    # Create fresh
    cursor.execute(f"CREATE DATABASE {DB_NAME}")
    print(f"Created database {DB_NAME}")

    cursor.close()
    conn.close()


def create_tables(conn):
    """Creates Customers, Products, and partitioned Orders tables.
    Orders table is partitioned by OrderID with 3 equal partitions."""
    cursor = conn.cursor()

    # Create Customers table
    cursor.execute(
        """
        CREATE TABLE Customers (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """
    )
    print("Created table 'Customers'")

    # Create Products table
    cursor.execute(
        """
        CREATE TABLE Products (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            price DECIMAL(10, 2) NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """
    )
    print("Created table 'Products'")

    # Create partitioned Orders table (partitioned by OrderID)
    cursor.execute(
        """
        CREATE TABLE Orders (
            OrderID INTEGER NOT NULL,
            CustomerId INTEGER NOT NULL,
            ProductId INTEGER NOT NULL,
            ProductQuantity INTEGER NOT NULL,
            TotalAmount DECIMAL(10, 2) NOT NULL,
            CreatedAt TIMESTAMP NOT NULL DEFAULT NOW()
        ) PARTITION BY RANGE (OrderID)
    """
    )
    print("Created partitioned parent table 'Orders'")

    # Create 3 partitions with equal splits
    boundary1 = NUM_ORDERS // 3
    boundary2 = (NUM_ORDERS * 2) // 3

    cursor.execute(
        f"""
        CREATE TABLE orders_p1 PARTITION OF Orders
        FOR VALUES FROM (0) TO ({boundary1})
    """
    )

    cursor.execute(
        f"""
        CREATE TABLE orders_p2 PARTITION OF Orders
        FOR VALUES FROM ({boundary1}) TO ({boundary2})
    """
    )

    cursor.execute(
        f"""
        CREATE TABLE orders_p3 PARTITION OF Orders
        FOR VALUES FROM ({boundary2}) TO ({NUM_ORDERS + 1})
    """
    )

    print(
        f"Created 3 order partitions: [0-{boundary1}), [{boundary1}-{boundary2}), [{boundary2}-{NUM_ORDERS + 1})"
    )

    conn.commit()
    cursor.close()


def populate_tables(conn):
    """Inserts data into Customers, Products, and Orders tables."""
    cursor = conn.cursor()

    # Insert Customers
    print(f"Inserting {NUM_CUSTOMERS} customers...")
    for i in range(1, NUM_CUSTOMERS + 1):
        cursor.execute(
            """
            INSERT INTO Customers (name, email)
            VALUES (%s, %s)
        """,
            (f"Customer {i}", f"customer{i}@example.com"),
        )
        # if i % 100 == 0:
        #     print(f"  Inserted {i}/{NUM_CUSTOMERS} customers")
    conn.commit()
    print(f"Successfully inserted {NUM_CUSTOMERS} customers")

    # Insert Products
    print(f"Inserting {NUM_PRODUCTS} products...")
    for i in range(1, NUM_PRODUCTS + 1):
        price = Decimal(random.uniform(10.0, 1000.0)).quantize(Decimal("0.01"))
        cursor.execute(
            """
            INSERT INTO Products (name, price)
            VALUES (%s, %s)
        """,
            (f"Product {i}", price),
        )
    conn.commit()
    print(f"Successfully inserted {NUM_PRODUCTS} products")

    # Insert Orders
    print(f"Inserting {NUM_ORDERS} orders...")
    for order_id in range(1, NUM_ORDERS + 1):
        customer_id = random.randint(1, NUM_CUSTOMERS)
        product_id = random.randint(1, NUM_PRODUCTS)
        quantity = random.randint(1, 10)

        # Get product price to calculate total
        cursor.execute("SELECT price FROM Products WHERE id = %s", (product_id,))
        product_price = cursor.fetchone()[0]
        total_amount = product_price * quantity

        cursor.execute(
            """
            INSERT INTO Orders (OrderID, CustomerId, ProductId, ProductQuantity, TotalAmount)
            VALUES (%s, %s, %s, %s, %s)
        """,
            (order_id, customer_id, product_id, quantity, total_amount),
        )

        # if order_id % 1000 == 0:
        #     print(f"  Inserted {order_id}/{NUM_ORDERS} orders")

    conn.commit()
    cursor.close()
    print(f"Successfully inserted {NUM_ORDERS} orders")


def main():
    """Orchestrates database initialization.
    Calls drop, create, partition, and populate functions."""
    print("=== Starting Database Initialization ===\n")

    # Step 1: Drop and create database
    drop_and_create_database()

    # Step 2: Connect to new database
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )

    # Step 3: Create tables (including partitioned Orders table)
    create_tables(conn)

    # Step 4: Populate with data
    populate_tables(conn)

    conn.close()
    print("\n=== Database Initialization Complete ===")


if __name__ == "__main__":
    main()
