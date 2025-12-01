import psycopg2
import sys

def check_databases():
    ports = [5432, 5433, 5434]
    config = {
        'dbname': 'tpch_db',
        'user': 'partitionuser',
        'password': 'partitionpass',
        'host': 'localhost'
    }
    
    for port in ports:
        try:
            cfg = config.copy()
            cfg['port'] = port
            conn = psycopg2.connect(**cfg)
            conn.close()
            print(f"Port {port} is running")
        except:
            print(f"Port {port} is NOT running")
            return False
    return True

def check_data():
    config = {
        'dbname': 'tpch_db',
        'user': 'partitionuser',
        'password': 'partitionpass',
        'host': 'localhost',
        'port': 5432
    }
    
    conn = psycopg2.connect(**config)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM orders")
    orders = cur.fetchone()[0]
    
    if orders > 0:
        print(f" Data loaded: {orders:,} orders found")
        cur.close()
        conn.close()
        return True
    else:
        print(f"No data found")
        cur.close()
        conn.close()
        return False

def main():
    checks = [
        ("Databases Running", check_databases),
        ("Data Loaded", check_data)
    ]
    
    for name, check_func in checks:
        if not check_func():
            print(f"{name} failed")


if __name__ == "__main__":
    main()