"""
Auto-generated database configuration
"""

COORDINATOR_CONFIG = {
    'dbname': 'tpch_db',
    'user': 'partitionuser',
    'password': 'partitionpass',  # Update if different
    'host': 'localhost',
    'port': 5432
}

def get_coordinator_config():
    return COORDINATOR_CONFIG.copy()

if __name__ == "__main__":
    print("Database Configuration:")
    print(f"  Host: {COORDINATOR_CONFIG['host']}")
    print(f"  Port: {COORDINATOR_CONFIG['port']}")
    print(f"  User: {COORDINATOR_CONFIG['user']}")
    print(f"  Database: {COORDINATOR_CONFIG['dbname']}")
