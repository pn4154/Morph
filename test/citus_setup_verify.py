#!/usr/bin/env python3
import psycopg2
import time

DB_CONFIG = {
    'dbname': 'tpch_db',
    'user': 'partitionuser',
    'password': 'partitionpass',
    'host': 'localhost',
    'port': 5432
}

def wait_for_database(max_attempts=30):
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(**DB_CONFIG, connect_timeout=3)
            conn.close()
            print("Database is ready")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_attempts}: Not ready yet...")
            time.sleep(2)
    
    print("Database did not become ready in time")
    return False

def verify_citus_setup():
    if not wait_for_database():
        return False
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print("CITUS CLUSTER VERIFICATION")
        
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"\nPostgreSQL Version:")
        print(f"  {version[:80]}...")
        
        cursor.execute("SELECT * FROM citus_version();")
        citus_version = cursor.fetchone()
        print(f"\nCitus Extension:")
        print(f"  Version: {citus_version[0]}")
        
        cursor.execute("SELECT * FROM citus_get_active_worker_nodes();")
        workers = cursor.fetchall()
        print(f"\nWorker Nodes: {len(workers)} active")
        for worker in workers:
            print(f"  - {worker[0]}:{worker[1]}")
        
        # Check extensions
        cursor.execute("""
            SELECT extname FROM pg_extension 
            WHERE extname IN ('citus', 'pg_stat_statements');
        """)
        extensions = cursor.fetchall()
        print(f"\nExtensions Installed:")
        for ext in extensions:
            print(f"  - {ext[0]}")
        
        # Check database
        cursor.execute("SELECT current_database();")
        db = cursor.fetchone()[0]
        print(f"\nCurrent Database: {db}")
        
        cursor.close()
        conn.close()
        
        print("CITUS CLUSTER IS READY")
        return True
        
    except Exception as e:
        print(f"\nError verifying Citus setup:")
        print(f"  {e}")
        return False

if __name__ == "__main__":
    success = verify_citus_setup()
    exit(0 if success else 1)