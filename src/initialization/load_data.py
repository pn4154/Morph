import psycopg2
import os

DB_CONFIG = {
    'dbname': 'tpch_db',
    'user': 'partitionuser',
    'password': 'partitionpass',
    'host': 'localhost',
    'port': 5432
}

# Get the project root directory (2 levels up from this file)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def load_table(cursor, table_name, file_path):
    with open(file_path, 'r') as f:
        cursor.copy_expert(f"COPY {table_name} FROM STDIN WITH DELIMITER '|'", f)
    print(f"Loaded {table_name}")


def main():
    print(f"Looking for data in: {DATA_DIR}")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    tables = [
        ('customer', 'customer.tbl'),
        ('part', 'part.tbl'),
        ('orders', 'orders.tbl'),
        ('lineitem', 'lineitem.tbl')
    ]
    
    for table_name, file_name in tables:
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(file_path):
            try:
                load_table(cur, table_name, file_path)
                conn.commit()
            except Exception as e:
                print(f"Error loading {table_name}: {e}")
                conn.rollback()
        else:
            print(f"File not found: {file_path}")
    
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()