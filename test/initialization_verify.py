import psycopg2

DB_CONFIG = {
    'dbname': 'tpch_db',
    'user': 'partitionuser',
    'password': 'partitionpass',
    'host': 'localhost',
    'port': 5432
}

def verify():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    tables = ['customer', 'orders', 'part', 'lineitem']
    
    print("\nData Verification:")
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        count = cur.fetchone()[0]
        print(f"{table}: {count:,} rows")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    verify()