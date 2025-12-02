#!/bin/bash
#
# Morph: Docker Compose Quick Start
# ==================================
# Use this script for local development with Docker Compose
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

case "${1:-start}" in
    start)
        log_info "Starting Morph with Docker Compose..."
        docker-compose up -d
        
        log_info "Waiting for cluster initialization..."
        sleep 30
        
        log_info "Loading TPC-H data..."
        pip install psycopg2-binary --quiet 2>/dev/null || true
        python3 scripts/generate_tpch_data.py \
            --scale 0.1 \
            --host localhost \
            --port 5432
        
        log_success "Morph is ready!"
        echo ""
        echo "Services:"
        echo "  - PostgreSQL: localhost:5432"
        echo "  - Prometheus: http://localhost:9090"
        echo "  - Grafana:    http://localhost:3000 (admin/admin)"
        ;;
    
    stop)
        log_info "Stopping Morph..."
        docker-compose down
        log_success "Done!"
        ;;
    
    clean)
        log_info "Cleaning up Morph..."
        docker-compose down -v
        log_success "Done!"
        ;;
    
    logs)
        docker-compose logs -f "${2:-coordinator}"
        ;;
    
    status)
        docker-compose ps
        ;;
    
    shell)
        docker-compose exec coordinator psql -U morph -d morphdb
        ;;
    
    *)
        echo "Usage: $0 {start|stop|clean|logs|status|shell}"
        exit 1
        ;;
esac
