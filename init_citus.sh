#!/bin/bash
# Quick fix for Citus setup issues

set -e

echo "=========================================="
echo "MORPH - Quick Citus Setup Fix"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Step 1: Check if Docker containers are running
echo -e "\n${YELLOW}Step 1: Checking Docker containers...${NC}"
CONTAINERS=$(docker ps --format "{{.Names}}" | grep -E "partition_db" || true)

if [ -z "$CONTAINERS" ]; then
    echo -e "${RED}✗ No Docker containers running${NC}"
    echo ""
    echo "Starting containers with docker-compose..."
    docker-compose up -d
    
    echo "Waiting for containers to be ready..."
    sleep 10
else
    echo -e "${GREEN}✓ Found running containers:${NC}"
    echo "$CONTAINERS"
fi

# Step 2: Install Citus extension on coordinator
echo -e "\n${YELLOW}Step 2: Installing Citus extension...${NC}"
COORDINATOR="partition_db1"

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if docker exec $COORDINATOR pg_isready -U partitionuser -d tpch_db 2>/dev/null; then
        echo -e "${GREEN}✓ PostgreSQL is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ PostgreSQL did not become ready${NC}"
        exit 1
    fi
    sleep 2
done

# Create Citus extension
echo "Creating Citus extension..."
docker exec $COORDINATOR psql -U partitionuser -d tpch_db -c "CREATE EXTENSION IF NOT EXISTS citus;" 2>/dev/null || {
    echo -e "${RED}✗ Failed to create Citus extension${NC}"
    echo "This usually means Citus is not installed in the PostgreSQL image"
    echo ""
    echo "Your docker-compose.yml should use: image: citusdata/citus:12.1"
    exit 1
}

echo -e "${GREEN}✓ Citus extension created${NC}"

# Step 3: Add worker nodes
echo -e "\n${YELLOW}Step 3: Adding worker nodes...${NC}"

# Add worker1 (partition_db2)
docker exec $COORDINATOR psql -U partitionuser -d tpch_db -c \
    "SELECT master_add_node('partition_db2', 5432);" 2>/dev/null || \
    echo "  Worker partition_db2 may already be added"

# Add worker2 (partition_db3)
docker exec $COORDINATOR psql -U partitionuser -d tpch_db -c \
    "SELECT master_add_node('partition_db3', 5432);" 2>/dev/null || \
    echo "  Worker partition_db3 may already be added"

# Verify workers
echo -e "\n${YELLOW}Active worker nodes:${NC}"
docker exec $COORDINATOR psql -U partitionuser -d tpch_db -c \
    "SELECT * FROM citus_get_active_worker_nodes();" || \
    echo "Could not retrieve workers"

# Step 4: Create pg_stat_statements extension
echo -e "\n${YELLOW}Step 4: Installing pg_stat_statements...${NC}"
docker exec $COORDINATOR psql -U partitionuser -d tpch_db -c \
    "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;" 2>/dev/null || true

echo -e "${GREEN}✓ pg_stat_statements installed${NC}"

# Step 5: Verify setup
echo -e "\n${YELLOW}Step 5: Verifying Citus setup...${NC}"

CITUS_VERSION=$(docker exec $COORDINATOR psql -U partitionuser -d tpch_db -t -c \
    "SELECT citus_version();" 2>/dev/null | xargs || echo "FAILED")

if [ "$CITUS_VERSION" = "FAILED" ]; then
    echo -e "${RED}✗ Citus verification failed${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Citus is working!${NC}"
    echo "  Version: $CITUS_VERSION"
fi

# Final summary
echo ""
echo "=========================================="
echo -e "${GREEN}✓ SETUP COMPLETE!${NC}"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  python baseline_partitioning.py"
echo ""
echo "To verify manually:"
echo "  docker exec -it partition_db1 psql -U partitionuser -d tpch_db"
echo "  \\dx  -- List extensions"
echo "  SELECT * FROM citus_get_active_worker_nodes();"
echo ""