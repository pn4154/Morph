#!/bin/bash
#
# Morph: Complete Setup and Deployment Script
# ============================================
# This script sets up the entire Morph infrastructure including:
# - Kubernetes cluster with Minikube
# - Citus PostgreSQL cluster (coordinator + 3 workers)
# - Prometheus monitoring
# - TPC-H schema and data
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROFILE_NAME="morph-cluster"
NAMESPACE="morph"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================
# Phase 1: Cleanup
# ============================================
cleanup() {
    log_info "Starting cleanup..."
    
    # Delete namespace if exists
    if kubectl get namespace $NAMESPACE &>/dev/null; then
        log_info "Deleting namespace $NAMESPACE..."
        kubectl delete namespace $NAMESPACE --ignore-not-found=true
        sleep 10
    fi
    
    # Stop and delete minikube cluster
    if minikube profile list 2>/dev/null | grep -q $PROFILE_NAME; then
        log_info "Stopping minikube cluster..."
        minikube stop -p $PROFILE_NAME 2>/dev/null || true
        
        log_info "Deleting minikube cluster..."
        minikube delete -p $PROFILE_NAME 2>/dev/null || true
    fi
    
    log_success "Cleanup complete!"
}

# ============================================
# Phase 2: Start Kubernetes Cluster
# ============================================
start_cluster() {
    log_info "Starting Kubernetes cluster..."
    
    minikube start \
        --nodes=3 \
        --cpus=2 \
        --memory=4096 \
        --disk-size=20g \
        --driver=docker \
        -p $PROFILE_NAME
    
    # Wait for nodes to be ready
    log_info "Waiting for nodes to be ready..."
    kubectl wait --for=condition=ready node --all --timeout=300s
    
    log_success "Kubernetes cluster started with $(kubectl get nodes --no-headers | wc -l) nodes"
    kubectl get nodes
}

# ============================================
# Phase 3: Deploy Citus Cluster
# ============================================
deploy_citus() {
    log_info "Deploying Citus cluster..."
    
    # Apply all Kubernetes manifests
    kubectl apply -f "$PROJECT_ROOT/kubernetes/citus-cluster.yaml"
    
    # Wait for coordinator
    log_info "Waiting for coordinator to be ready..."
    kubectl wait --for=condition=ready pod -l app=citus-coordinator -n $NAMESPACE --timeout=300s
    
    # Wait for workers
    log_info "Waiting for workers to be ready..."
    kubectl wait --for=condition=ready pod -l app=citus-worker -n $NAMESPACE --timeout=300s
    
    log_success "Citus cluster deployed!"
    kubectl get pods -n $NAMESPACE
}

# ============================================
# Phase 4: Initialize Citus Extensions
# ============================================
init_extensions() {
    log_info "Initializing extensions on workers..."
    
    # Get coordinator pod
    COORDINATOR_POD=$(kubectl get pods -n $NAMESPACE -l app=citus-coordinator -o jsonpath='{.items[0].metadata.name}')
    
    # Initialize extensions on each worker first
    for i in 0 1 2; do
        log_info "Initializing extensions on worker-$i..."
        kubectl exec -n $NAMESPACE citus-worker-$i -- \
            psql -U morph -d morphdb -c "CREATE EXTENSION IF NOT EXISTS citus;" 2>/dev/null || true
        kubectl exec -n $NAMESPACE citus-worker-$i -- \
            psql -U morph -d morphdb -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;" 2>/dev/null || true
    done
    
    # Initialize extensions on coordinator
    log_info "Initializing extensions on coordinator..."
    kubectl exec -n $NAMESPACE $COORDINATOR_POD -- \
        psql -U morph -d morphdb -c "CREATE EXTENSION IF NOT EXISTS citus;" 2>/dev/null || true
    kubectl exec -n $NAMESPACE $COORDINATOR_POD -- \
        psql -U morph -d morphdb -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;" 2>/dev/null || true
    
    log_success "Extensions initialized!"
}

# ============================================
# Phase 5: Register Worker Nodes
# ============================================
register_workers() {
    log_info "Registering worker nodes..."
    
    COORDINATOR_POD=$(kubectl get pods -n $NAMESPACE -l app=citus-coordinator -o jsonpath='{.items[0].metadata.name}')
    
    # Set coordinator host
    kubectl exec -n $NAMESPACE $COORDINATOR_POD -- \
        psql -U morph -d morphdb -c "SELECT citus_set_coordinator_host('citus-coordinator.morph.svc.cluster.local');"
    
    # Add workers
    for i in 0 1 2; do
        log_info "Adding worker-$i..."
        kubectl exec -n $NAMESPACE $COORDINATOR_POD -- \
            psql -U morph -d morphdb -c \
            "SELECT citus_add_node('citus-worker-$i.citus-worker.morph.svc.cluster.local', 5432);" 2>/dev/null || \
            log_warning "Worker-$i may already be registered"
    done
    
    # Verify workers
    log_info "Verifying worker registration..."
    kubectl exec -n $NAMESPACE $COORDINATOR_POD -- \
        psql -U morph -d morphdb -c "SELECT * FROM citus_get_active_worker_nodes();"
    
    log_success "Workers registered!"
}

# ============================================
# Phase 6: Initialize Schema
# ============================================
init_schema() {
    log_info "Initializing TPC-H schema..."
    
    COORDINATOR_POD=$(kubectl get pods -n $NAMESPACE -l app=citus-coordinator -o jsonpath='{.items[0].metadata.name}')
    
    # Copy and execute schema script
    kubectl cp "$PROJECT_ROOT/scripts/init_schema.sql" $NAMESPACE/$COORDINATOR_POD:/tmp/init_schema.sql
    kubectl exec -n $NAMESPACE $COORDINATOR_POD -- \
        psql -U morph -d morphdb -f /tmp/init_schema.sql
    
    log_success "Schema initialized!"
}

# ============================================
# Phase 7: Load TPC-H Data
# ============================================
load_data() {
    log_info "Loading TPC-H data..."
    
    # Port-forward coordinator for data loading
    log_info "Starting port-forward..."
    kubectl port-forward -n $NAMESPACE svc/citus-coordinator 5432:5432 &
    PF_PID=$!
    sleep 5
    
    # Install Python dependencies
    pip install psycopg2-binary --quiet 2>/dev/null || pip install psycopg2 --quiet
    
    # Run data generator
    python3 "$PROJECT_ROOT/scripts/generate_tpch_data.py" \
        --scale 0.1 \
        --host localhost \
        --port 5432 \
        --database morphdb \
        --user morph \
        --password 'MorphSecurePass123!'
    
    # Stop port-forward
    kill $PF_PID 2>/dev/null || true
    
    log_success "TPC-H data loaded!"
}

# ============================================
# Phase 8: Verify Setup
# ============================================
verify_setup() {
    log_info "Verifying setup..."
    
    COORDINATOR_POD=$(kubectl get pods -n $NAMESPACE -l app=citus-coordinator -o jsonpath='{.items[0].metadata.name}')
    
    echo ""
    log_info "Checking shard distribution..."
    kubectl exec -n $NAMESPACE $COORDINATOR_POD -- \
        psql -U morph -d morphdb -c \
        "SELECT nodename, COUNT(*) as shard_count FROM pg_dist_shard_placement GROUP BY nodename ORDER BY nodename;"
    
    echo ""
    log_info "Checking table row counts..."
    kubectl exec -n $NAMESPACE $COORDINATOR_POD -- \
        psql -U morph -d morphdb -c \
        "SELECT 'customer' as table_name, COUNT(*) FROM customer
         UNION ALL SELECT 'orders', COUNT(*) FROM orders
         UNION ALL SELECT 'lineitem', COUNT(*) FROM lineitem
         UNION ALL SELECT 'part', COUNT(*) FROM part
         UNION ALL SELECT 'supplier', COUNT(*) FROM supplier;"
    
    echo ""
    log_info "Testing sample query..."
    kubectl exec -n $NAMESPACE $COORDINATOR_POD -- \
        psql -U morph -d morphdb -c \
        "SELECT l_returnflag, COUNT(*) FROM lineitem GROUP BY l_returnflag LIMIT 5;"
    
    log_success "Setup verification complete!"
}

# ============================================
# Phase 9: Setup Prometheus Access
# ============================================
setup_monitoring() {
    log_info "Setting up monitoring access..."
    
    # Get Prometheus NodePort
    PROMETHEUS_PORT=$(kubectl get svc -n $NAMESPACE prometheus -o jsonpath='{.spec.ports[0].nodePort}')
    MINIKUBE_IP=$(minikube ip -p $PROFILE_NAME)
    
    echo ""
    log_success "Prometheus accessible at: http://$MINIKUBE_IP:$PROMETHEUS_PORT"
    
    # Instructions for database access
    echo ""
    log_info "To access the database locally, run:"
    echo "  kubectl port-forward -n $NAMESPACE svc/citus-coordinator 5432:5432"
    echo ""
    log_info "Then connect with:"
    echo "  psql -h localhost -U morph -d morphdb"
}

# ============================================
# Main
# ============================================
main() {
    echo ""
    echo "============================================"
    echo "    Morph: Complete Setup Script"
    echo "============================================"
    echo ""
    
    case "${1:-all}" in
        cleanup)
            cleanup
            ;;
        cluster)
            start_cluster
            ;;
        deploy)
            deploy_citus
            ;;
        init)
            init_extensions
            register_workers
            init_schema
            ;;
        data)
            load_data
            ;;
        verify)
            verify_setup
            ;;
        all)
            cleanup
            start_cluster
            deploy_citus
            sleep 30  # Wait for pods to stabilize
            init_extensions
            register_workers
            init_schema
            load_data
            verify_setup
            setup_monitoring
            ;;
        *)
            echo "Usage: $0 {cleanup|cluster|deploy|init|data|verify|all}"
            exit 1
            ;;
    esac
    
    echo ""
    log_success "Done!"
}

main "$@"
