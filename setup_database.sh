#!/bin/bash
# Setup script that works with your existing docker-compose configuration

set -e

echo "========================================================================"
echo "MORPH DATABASE SETUP (Using Existing Configuration)"
echo "========================================================================"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Detect container names
echo -e "\n${YELLOW}Detecting running containers...${NC}"
CONTAINERS=$(docker ps --format "{{.Names}}" | grep -E "partition_db|citus" || true)

if [ -z "$CONTAINERS" ]; then
    echo -e "${RED}✗ No database containers found${NC}"
    echo "Run: docker-compose up -d"
    exit 1
fi

echo -e "${GREEN}✓ Found containers:${NC}"
echo "$CONTAINERS"

# Identify coordinator (usually the first one or port 5432)
COORDINATOR=$(docker ps --format "{{.Names}}" --filter "publish=5432" | head -n 1)

if [ -z "$COORDINATOR" ]; then
    # Fallback: use partition_db1 or first container
    COORDINATOR=$(echo "$CONTAINERS" | grep "partition_db1" || echo "$CONTAINERS" | head -n 1)
fi

echo -e "\n${GREEN}✓ Using coordinator: $COORDINATOR${NC}"

# Wait for database to be ready
echo -e "\n${YELLOW}Waiting for database to be ready...${NC}"
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if docker exec $COORDINATOR pg_isready -U postgres 2>/dev/null || \
       docker exec $COORDINATOR pg_isready -U partitionuser 2>/dev/null; then
        echo -e "${GREEN}✓ Database is ready!${NC}"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo -n "."
    sleep 2
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo -e "\n${RED}✗ Database did not become ready${NC}"
    exit 1
fi

# Determine database user and name
echo -e "\n${YELLOW}Detecting database configuration...${NC}"

# Try different user/database combinations
for USER in partitionuser postgres; do
    for DB in tpch_db postgres; do
        if docker exec $COORDINATOR psql -U $USER -d $DB -c "SELECT 1;" &>/dev/null; then
            DB_USER=$USER
            DB_NAME=$DB
            echo -e "${GREEN}✓ Connected with user: $DB_USER, database: $DB_NAME${NC}"
            break 2
        fi
    done
done

if [ -z "$DB_USER" ]; then
    echo -e "${RED}✗ Could not connect to database${NC}"
    exit 1
fi

# Check if we have Citus or plain PostgreSQL
echo -e "\n${YELLOW}Checking database type...${NC}"
IS_CITUS=$(docker exec $COORDINATOR psql -U $DB_USER -d $DB_NAME -t -c "SELECT 1 FROM pg_extension WHERE extname='citus';" 2>/dev/null | xargs || echo "0")

if [ "$IS_CITUS" = "1" ]; then
    echo -e "${GREEN}✓ Citus is already installed${NC}"
else
    echo -e "${YELLOW}⚠ Citus not installed. Attempting to install...${NC}"
    
    # Try to install Citus
    docker exec $COORDINATOR psql -U $DB_USER -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS citus;" 2>/dev/null || true
    
    IS_CITUS=$(docker exec $COORDINATOR psql -U $DB_USER -d $DB_NAME -t -c "SELECT 1 FROM pg_extension WHERE extname='citus';" 2>/dev/null | xargs || echo "0")
    
    if [ "$IS_CITUS" = "1" ]; then
        echo -e "${GREEN}✓ Citus installed successfully${NC}"
    else
        echo -e "${YELLOW}⚠ Could not install Citus (using plain PostgreSQL)${NC}"
    fi
fi

# Install pg_stat_statements
echo -e "\n${YELLOW}Installing pg_stat_statements...${NC}"
docker exec $COORDINATOR psql -U $DB_USER -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;" 2>/dev/null || true

# If Citus is available, set up workers
if [ "$IS_CITUS" = "1" ]; then
    echo -e "\n${YELLOW}Setting up Citus worker nodes...${NC}"
    
    # Get all database containers except coordinator
    WORKERS=$(docker ps --format "{{.Names}}" | grep -E "partition_db|citus" | grep -v "$COORDINATOR" || true)
    
    if [ -n "$WORKERS" ]; then
        for WORKER in $WORKERS; do
            # Get worker port mapping
            WORKER_PORT=$(docker port $WORKER 5432 2>/dev/null | cut -d: -f2 || echo "5432")
            
            # Use container name as hostname (Docker networking)
            echo "  Adding worker: $WORKER"
            docker exec $COORDINATOR psql -U $DB_USER -d $DB_NAME -c \
                "SELECT master_add_node('$WORKER', 5432);" 2>/dev/null || \
                echo "    (Worker may already be added)"
        done
    fi
    
    # Show active workers
    echo -e "\n${YELLOW}Active worker nodes:${NC}"
    docker exec $COORDINATOR psql -U $DB_USER -d $DB_NAME -c \
        "SELECT * FROM citus_get_active_worker_nodes();" 2>/dev/null || \
        echo "Could not retrieve worker nodes"
fi

# Create tpch_db if it doesn't exist
echo -e "\n${YELLOW}Ensuring tpch_db exists...${NC}"
docker exec $COORDINATOR psql -U $DB_USER -d postgres -c "CREATE DATABASE tpch_db;" 2>/dev/null || echo "  Database already exists"

# Verify connection to tpch_db
if docker exec $COORDINATOR psql -U $DB_USER -d tpch_db -c "SELECT 1;" &>/dev/null; then
    echo -e "${GREEN}✓ tpch_db is accessible${NC}"
    
    # Install extensions in tpch_db too
    docker exec $COORDINATOR psql -U $DB_USER -d tpch_db -c "CREATE EXTENSION IF NOT EXISTS citus;" 2>/dev/null || true
    docker exec $COORDINATOR psql -U $DB_USER -d tpch_db -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;" 2>/dev/null || true
fi

# Display connection info
echo -e "\n========================================================================"
echo -e "${GREEN}✓ SETUP COMPLETE!${NC}"
echo "========================================================================"
echo -e "\nDatabase connection details:"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  User: $DB_USER"
echo "  Database: tpch_db"
echo ""
echo "Container: $COORDINATOR"
echo ""

# Generate Python config
cat > config.py << PYEOF
"""
Auto-generated database configuration
"""

COORDINATOR_CONFIG = {
    'dbname': 'tpch_db',
    'user': '$DB_USER',
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
PYEOF

echo "Generated config.py with connection details"
echo ""
echo "Next steps:"
echo "  1. Run: python verify_citus_setup.py"
echo "  2. Run: python setup_schema.py"
echo "  3. Generate and load TPC-H data"
echo "  4. Run: python baseline_partitioning.py"
echo ""
echo "Useful commands:"
echo "  - Connect: docker exec -it $COORDINATOR psql -U $DB_USER -d tpch_db"
echo "  - Logs:    docker logs $COORDINATOR"
echo "  - Stop:    docker-compose down"
echo ""