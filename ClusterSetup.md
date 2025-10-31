
## 1. Environment Setup

### Phase 1: Complete Cleanup

```bash
echo "=========================================="
echo "Complete Environment Cleanup"
echo "=========================================="

# Delete all resources in morph namespace
kubectl delete namespace morph

# Wait for namespace to be deleted
echo "Waiting for namespace deletion..."
sleep 10

# Stop minikube cluster
minikube stop -p morph-cluster

# Delete minikube cluster completely
minikube delete -p morph-cluster

# Clean up local files
cd ~
rm -rf morph

echo "Cleanup complete!"
```


### Phase 2: Fresh Setup with Fixed Scripts
Step 1: Create Project Structure
```
mkdir -p ~/morph/{kubernetes,scripts,data}
cd ~/morph
echo "Project structure created at ~/morph"
```
Step 2: Start Fresh Kubernetes Cluster
```
minikube start \
  --nodes=3 \
  --cpus=2 \
  --memory=4096 \
  --disk-size=20g \
  --driver=docker \
  -p morph-cluster

kubectl get nodes
echo "Kubernetes cluster ready!"
```
Step 3: Create Namespace and Secrets
```
kubectl create namespace morph

kubectl create secret generic postgres-secrets \
  --from-literal=password='MorphSecurePass123!' \
  -n morph

echo "Namespace and secrets created!"
```
Step 4: Create Storage
```
cat > kubernetes/storage.yaml << 'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: citus-coordinator-pvc
  namespace: morph
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF

kubectl apply -f kubernetes/storage.yaml
```

Step 5: Deploy Citus Coordinator & Workers
Deployed coordinator and worker nodes with custom pg_hba.conf fixes to allow connections.
Enabled required extensions (citus, pg_stat_statements) on workers first, then on coordinator.
Verification scripts:
```
kubectl wait --for=condition=ready pod -l app=citus-coordinator -n morph --timeout=180s
kubectl wait --for=condition=ready pod -l app=citus-worker -n morph --timeout=300s
```
Step 6: Initialize Citus Cluster
```
Added all workers to coordinator:
SELECT citus_add_node('citus-worker-0.citus-worker.morph.svc.cluster.local', 5432);
SELECT citus_add_node('citus-worker-1.citus-worker.morph.svc.cluster.local', 5432);
SELECT citus_add_node('citus-worker-2.citus-worker.morph.svc.cluster.local', 5432);

-- Verify
SELECT * FROM citus_get_active_worker_nodes();
Enabled extensions on coordinator after workers were ready.
Tested connectivity to all workers:
for i in 0 1 2; do
  kubectl exec -n morph $COORDINATOR_POD -- \
    psql -U morph -h citus-worker-$i.citus-worker.morph.svc.cluster.local -d morphdb -c "SELECT 'Connected' as status;"
done
```
2. Shard Management
Coordinator temporarily holding shards:
Even after rebalance_table_shards_to_workers(), the coordinator may retain some shards due to Citus metadata behavior.
Final shard verification:
```
SELECT nodename, count(*) AS shard_count
FROM pg_dist_shard_placement
JOIN pg_dist_shard USING (shardid)
GROUP BY nodename
ORDER BY nodename;
Example Output:
nodename	shard_count
coordinator	18
worker-0	18
worker-1	18
worker-2	18
```
Coordinator still holds some shards. This is acceptable for RL experiments as most data resides on workers.

Important: Always rebalance before any table/schema drop operations.

4. Database Initialization & Schema
Copied schema scripts to coordinator:
```
kubectl cp scripts/create_schema.sql morph/$COORDINATOR_POD:/tmp/create_schema.sql
kubectl exec -n morph $COORDINATOR_POD -- \
  psql -U morph -d morphdb -f /tmp/create_schema.sql
```


6. Issues & Debugging
Dropping schema too early: Dropped public schema caused cache lookup failures.

Solution: Drop tables individually after shards have been moved to workers.

Coordinator shard retention: Coordinator still retains some shards after rebalancing.
Acceptable for experiments as long as worker nodes hold majority of data.

Extension setup order: Always enable citus and pg_stat_statements on workers before coordinator.

Command errors in Zsh: Use quotes and avoid # inline comments in shell commands.

8. Notes for Reproducibility
Always perform complete environment cleanup before starting fresh.
Enable extensions on workers first, then coordinator.
Rebalance shards before any schema or table deletion.
Use the provided reset_citus.sh script for automation.
