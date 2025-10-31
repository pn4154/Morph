
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
Deployed coordinator and worker nodes with custom pg_hba.conf to allow connections.

Enabled required extensions (citus, pg_stat_statements) on workers first, then on coordinator.

Verification scripts:
```
kubectl wait --for=condition=ready pod -l app=citus-coordinator -n morph --timeout=180s
kubectl wait --for=condition=ready pod -l app=citus-worker -n morph --timeout=300s
```
Notes / Issues:

The coordinator must preload extensions citus and pg_stat_statements.

Custom pg_hba.conf required for trust authentication across pods.

Step 7: Initialize Citus Cluster
SQL Initialization (scripts/init_citus.sql)
```
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS citus;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Add worker nodes
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM citus_get_active_worker_nodes() WHERE node_name='citus-worker-0.citus-worker.morph.svc.cluster.local') THEN
        PERFORM citus_add_node('citus-worker-0.citus-worker.morph.svc.cluster.local', 5432);
    END IF;
    -- Repeat for worker-1 and worker-2
END $$;

-- Verify workers
SELECT * FROM citus_get_active_worker_nodes();
```
Worker Extension Setup:
```
for i in 0 1 2; do
  kubectl exec -n morph citus-worker-$i -- \
    psql -U morph -d morphdb -c "CREATE EXTENSION IF NOT EXISTS citus;"
  kubectl exec -n morph citus-worker-$i -- \
    psql -U morph -d morphdb -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"
done
```

Coordinator Initialization:

```
COORDINATOR_POD=$(kubectl get pods -n morph -l app=citus-coordinator -o jsonpath='{.items[0].metadata.name}')
kubectl cp scripts/init_citus.sql morph/$COORDINATOR_POD:/tmp/init_citus.sql
kubectl exec -n morph $COORDINATOR_POD -- \
  psql -U morph -d morphdb -f /tmp/init_citus.sql
```

Connectivity Check:

```
for i in 0 1 2; do
  kubectl exec -n morph $COORDINATOR_POD -- \
    psql -U morph -h citus-worker-$i.citus-worker.morph.svc.cluster.local -d morphdb -c "SELECT 'Connected' as status;"
done
```

### Phase 3: Shard Management & Debugging

Issues Faced-

1. Dropping schema too early caused errors:

ERROR: cache lookup failed for pg_dist_local_group, called too early?

Fix: Drop tables individually after moving shards to workers, do not drop public schema while shards exist on the coordinator.



3. Extension setup errors:

Always ensure citus and pg_stat_statements are created on workers first, then on the coordinator.

Correct Shard Rebalancing Flow

-> SELECT rebalance_table_shards_to_workers();

-> Verifies final shard distribution only on worker nodes.

-> Ensures a clean worker-only state before experiments.

-> Verification of Final Shard Distribution
```
SELECT nodename, count(*) AS shard_count
FROM pg_dist_shard_placement
JOIN pg_dist_shard USING (shardid)
GROUP BY nodename
ORDER BY nodename;

Expected output:

nodename	shard_count
worker-0	18
worker-1	18
worker-2	18
coordinator	0
```

#### Confirm worker pods are ready:
```
kubectl get pods -n morph -l app=citus-worker -o wide
kubectl get svc -n morph citus-worker

You should see:
3 worker pods with STATUS: Running
citus-worker service with CLUSTER-IP: None
```

#### Test DNS and Connectivity:
```
kubectl run -it dns-debug \
  --image=busybox:1.35 \
  --restart=Never \
  -n morph \
  --rm \
  --command -- nslookup citus-worker-0.citus-worker.morph.svc.cluster.local

✅ You should see output like this if DNS works:
Server:    10.96.0.10
Address 1: 10.96.0.10 kube-dns.kube-system.svc.cluster.local

Name:      citus-worker-0.citus-worker.morph.svc.cluster.local
Address 1: 10.244.1.23 citus-worker-0.citus-worker.morph.svc.cluster.local
```

✅ What This Confirms

Cluster DNS (CoreDNS) is working correctly.

Headless Service (citus-worker) is resolving each worker pod hostname properly.

The worker pod citus-worker-0 has the cluster-internal IP 10.244.2.3.

So the coordinator can theoretically reach the workers by hostname

citus-worker-0.citus-worker.morph.svc.cluster.local.

#### Check if the coordinator has Citus enabled
```
COORDINATOR_POD=$(kubectl get pods -n morph -l app=citus-coordinator -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n morph $COORDINATOR_POD -- psql -U morph -d morphdb -c "CREATE EXTENSION
```

#### Test direct connection from coordinator to worker
This ensures the coordinator can reach the worker over TCP:
```
kubectl exec -n morph $COORDINATOR_POD -- \
  psql -U morph -d morphdb -h citus-worker-0.citus-worker.morph.svc.cluster.local -c "SELECT 'Connected to worker-0' as status;"
Expected output:
     status
---------------------
 Connected to worker-0
```

🧩 What This Confirms

✅ Network + DNS:

The coordinator can resolve and reach the worker hostname correctly over the cluster network.

✅ PostgreSQL authentication:

Your custom pg_hba.conf trust rules are working — no password or host restrictions blocked the connection.

✅ Citus binaries:

Both coordinator and workers are running compatible citus versions (so the connection layer recognizes each other).


#### Register Workers (manually, to confirm)

✅ Step 1: Set the Coordinator Host

Run this (inside your coordinator pod):
```
kubectl exec -n morph $COORDINATOR_POD -- \
  psql -U morph -d morphdb -c "SELECT citus_set_coordinator_host('citus-coordinator.morph.svc.cluster.local');"
Expected output:
 citus_set_coordinator_host 
----------------------------
 
(1 row)
```
✅ Step 2: Add Workers Again

Now re-run the add commands:
```
kubectl exec -n morph $COORDINATOR_POD -- \
  psql -U morph -d morphdb -c "SELECT citus_add_node('citus-worker-0.citus-worker.morph.svc.cluster.local', 5432);"

kubectl exec -n morph $COORDINATOR_POD -- \
  psql -U morph -d morphdb -c "SELECT citus_add_node('citus-worker-1.citus-worker.morph.svc.cluster.local', 5432);"

kubectl exec -n morph $COORDINATOR_POD -- \
  psql -U morph -d morphdb -c "SELECT citus_add_node('citus-worker-2.citus-worker.morph.svc.cluster.local', 5432);"
```

✅ Step 3: Verify Cluster Setup

Finally, check that the nodes are registered:
```
kubectl exec -n morph $COORDINATOR_POD -- \
  psql -U morph -d morphdb -c "SELECT * FROM citus_get_active_worker_nodes();"

You should now see output like:
                 node_name                                  | node_port | nodeid
------------------------------------------------------------+-----------+--------
 citus-worker-0.citus-worker.morph.svc.cluster.local        |      5432 |      1
 citus-worker-1.citus-worker.morph.svc.cluster.local        |      5432 |      2
 citus-worker-2.citus-worker.morph.svc.cluster.local        |      5432 |      3
```



8. Notes for Reproducibility:

Always perform complete environment cleanup before starting fresh.

Enable extensions on workers first, then coordinator.

Rebalance shards before any schema or table deletion.

Use the provided reset_citus.sh script for automation.

If you run the schema before the workers are registered, Citus defaults to creating all shards on the coordinator. 
When workers are visible, you have to rebalance existing tables


