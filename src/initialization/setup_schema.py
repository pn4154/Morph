import psycopg2
from psycopg2 import sql

DB_CONFIG = {
    'dbname': 'tpch_db',
    'user': 'partitionuser',
    'password': 'partitionpass',
    'host': 'localhost',
    'port': 5432
}

SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS customer (
    c_custkey INTEGER ,
    c_name VARCHAR(25),
    c_address VARCHAR(40),
    c_nationkey INTEGER,
    c_phone CHAR(15),
    c_acctbal DECIMAL(15,2),
    c_mktsegment CHAR(10),
    c_comment VARCHAR(117)
);

CREATE TABLE IF NOT EXISTS orders (
    o_orderkey INTEGER ,
    o_custkey INTEGER,
    o_orderstatus CHAR(1),
    o_totalprice DECIMAL(15,2),
    o_orderdate DATE,
    o_orderpriority CHAR(15),
    o_clerk CHAR(15),
    o_shippriority INTEGER,
    o_comment VARCHAR(79)
);

CREATE TABLE IF NOT EXISTS part (
    p_partkey INTEGER ,
    p_name VARCHAR(55),
    p_mfgr CHAR(25),
    p_brand CHAR(10),
    p_type VARCHAR(25),
    p_size INTEGER,
    p_container CHAR(10),
    p_retailprice DECIMAL(15,2),
    p_comment VARCHAR(23)
);

CREATE TABLE IF NOT EXISTS lineitem (
    l_orderkey INTEGER,
    l_partkey INTEGER,
    l_suppkey INTEGER,
    l_linenumber INTEGER,
    l_quantity DECIMAL(15,2),
    l_extendedprice DECIMAL(15,2),
    l_discount DECIMAL(15,2),
    l_tax DECIMAL(15,2),
    l_returnflag CHAR(1),
    l_linestatus CHAR(1),
    l_shipdate DATE,
    l_commitdate DATE,
    l_receiptdate DATE,
    l_shipinstruct CHAR(25),
    l_shipmode CHAR(10),
    l_comment VARCHAR(44)
);

CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
"""

def setup_schema():
    try:
        print("Connecting to Citus coordinator (port 5432)")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        cur.execute(SQL_SCHEMA)
        conn.commit()
        
        cur.execute("""
            SELECT extname, extversion 
            FROM pg_extension 
            WHERE extname IN ('citus', 'pg_stat_statements')
            ORDER BY extname;
        """)
        extensions = cur.fetchall()
        
        print("\nSchema created successfully on coordinator!")
        for ext_name, ext_version in extensions:
            print(f"{ext_name}: {ext_version}")
        
        # Check if worker nodes are configured
        cur.execute("SELECT * FROM citus_get_active_worker_nodes();")
        workers = cur.fetchall()
        
        if workers:
            print(f"\nFound {len(workers)} active worker node(s):")
            for worker in workers:
                print(f"{worker[0]}:{worker[1]}")
        else:
            print("\nNo worker nodes configured")
\
        cur.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY tablename;
        """)
        tables = cur.fetchall()
        
        print(f"\nCreated {len(tables)} table(s):")
        for table in tables:
            print(f"  - {table[0]}")
        
        cur.close()
        conn.close()

    except Exception as e:
        print(f"\n✗ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    setup_schema()