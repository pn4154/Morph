# Morph: AI-Driven Adaptive Data Partitioning for Big Data Systems

**Learning-Based Optimization for Dynamic Workloads**

Morph is a reinforcement learning-based system that dynamically optimizes data partitioning in distributed PostgreSQL databases (Citus). It uses Proximal Policy Optimization (PPO) to learn optimal shard placement strategies based on workload patterns.

## Features

- **Reinforcement Learning Agent**: PPO-based agent with hybrid action space (discrete operations + continuous parameters)
- **Citus Integration**: Full integration with Citus distributed PostgreSQL
- **TPC-H Benchmark**: Standard database benchmark for evaluation
- **Multiple Action Types**: NO_OP, MOVE_PARTITION, SPLIT_PARTITION, MERGE_PARTITIONS, REBALANCE_PARTITION, MOVE_DATA_RANGE
- **Prometheus Monitoring**: Real-time performance metrics collection
- **Baseline Comparisons**: Evaluation against hash, round-robin, and workload-aware strategies

## Project Structure

```
morph/
├── kubernetes/              # Kubernetes deployment manifests
│   └── citus-cluster.yaml   # Full Citus cluster configuration
├── rl_agent/                # Reinforcement learning components
│   ├── __init__.py
│   ├── citus_env.py         # Gymnasium environment for Citus
│   ├── ppo_agent.py         # PPO implementation with hybrid actions
│   └── train.py             # Training script
├── scripts/                 # Setup and utility scripts
│   ├── setup.sh             # Complete setup automation
│   ├── init_schema.sql      # TPC-H schema for Citus
│   └── generate_tpch_data.py # TPC-H data generator
├── evaluation/              # Evaluation and benchmarking
│   └── evaluate.py          # Baseline comparison framework
├── configs/                 # Configuration files
├── data/                    # Data directory
├── docs/                    # Documentation
└── requirements.txt         # Python dependencies
```

## Quick Start

### Prerequisites

- Docker
- Minikube
- kubectl
- Python 3.9+
- 16GB RAM recommended

### 1. Setup Infrastructure

```bash
# Clone and navigate to project
cd morph

# Make setup script executable
chmod +x scripts/setup.sh

# Run complete setup (cluster, Citus, schema, data)
./scripts/setup.sh all
```

This will:
1. Clean up any existing Morph resources
2. Start a 3-node Minikube cluster
3. Deploy Citus coordinator and 3 workers
4. Initialize extensions and register workers
5. Create TPC-H schema with distributed tables
6. Load TPC-H benchmark data (0.1 scale factor)

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the RL Agent

```bash
# With simulated environment (no database required)
python rl_agent/train.py --simulate --total-timesteps 50000

# With actual Citus cluster
kubectl port-forward -n morph svc/citus-coordinator 5432:5432 &
python rl_agent/train.py \
    --coordinator-host localhost \
    --coordinator-port 5432 \
    --total-timesteps 100000
```

### 4. Evaluate Against Baselines

```bash
python evaluation/evaluate.py \
    --host localhost \
    --port 5432 \
    --duration 60 \
    --output-dir ./evaluation_results
```

## Architecture

### RL Environment

The `MorphEnv` class implements a Gymnasium environment with:

**Observation Space** (14 dimensions):
- Shard distribution per node (normalized)
- Node sizes
- Latency statistics (mean, p50, p95, p99)
- Workload features (read ratio, scan ratio, skew, time)

**Action Space** (Hybrid):
- Discrete: 6 action types (NO_OP, MOVE_PARTITION, SPLIT_PARTITION, MERGE_PARTITIONS, REBALANCE_PARTITION, MOVE_DATA_RANGE)
- Continuous: 4 parameters (shard_idx, target_node, split_point, merge_target)

**Reward Function**:
- Primary: Latency improvement over baseline
- Secondary: Load balance across nodes
- Penalties: Action cost, failed operations

### PPO Agent

The PPO implementation includes:
- Shared feature extractor (MLP with LayerNorm)
- Dual actor heads (discrete + continuous)
- Value critic head
- GAE advantage estimation
- Clipped objective with entropy regularization

### Citus Operations

Supported operations via Citus:
- `citus_move_shard_placement`: Move shards between workers
- `rebalance_table_shards`: Automatic rebalancing
- Custom implementations for split/merge (PostgreSQL-level)

## Configuration

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 3e-4 | Adam learning rate |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.95 | GAE lambda |
| clip_epsilon | 0.2 | PPO clip range |
| n_epochs | 10 | PPO epochs per update |
| n_steps | 2048 | Steps per rollout |
| batch_size | 64 | Mini-batch size |

### Environment Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_workers | 3 | Number of Citus workers |
| episode_length | 100 | Steps per episode |
| workload_type | mixed | oltp, olap, or mixed |

## Kubernetes Deployment

The Kubernetes deployment includes:

- **Citus Coordinator**: StatefulSet with 1 replica, 2GB memory
- **Citus Workers**: StatefulSet with 3 replicas, 1GB each
- **Prometheus**: Metrics collection
- **PostgreSQL Exporter**: Database metrics

Access services:
```bash
# Database
kubectl port-forward -n morph svc/citus-coordinator 5432:5432

# Prometheus
minikube service prometheus -n morph -p morph-cluster
```

## TPC-H Benchmark

The project uses TPC-H for evaluation:

| Table | Rows (SF=0.1) | Distribution |
|-------|---------------|--------------|
| customer | 15,000 | Hash by c_custkey |
| orders | 150,000 | Hash by o_orderkey |
| lineitem | ~600,000 | Hash by l_orderkey (colocated with orders) |
| part | 20,000 | Hash by p_partkey |
| supplier | 1,000 | Hash by s_suppkey |
| nation | 25 | Reference table |
| region | 5 | Reference table |

## Development

### Running Tests

```bash
# Test PPO agent
python -m pytest rl_agent/test_ppo.py -v

# Test environment
python -c "from rl_agent import MorphEnvFlat; print('OK')"
```

### Code Style

```bash
black rl_agent/
flake8 rl_agent/
mypy rl_agent/
```

## Results

Preliminary results from training show:
- RL agent learns to adjust partition boundaries based on workload skew
- Improved latency compared to static hash partitioning for skewed workloads
- Effective load balancing across worker nodes

See `evaluation_results/` for detailed benchmark comparisons.

## Roadmap

- [x] Phase 1: Infrastructure setup (Citus, Kubernetes, TPC-H)
- [x] Phase 2: Basic RL agent (PPO, NO_OP, MOVE_PARTITION)
- [ ] Phase 3: Extended operations (SPLIT, MERGE)
- [ ] Phase 4: Comprehensive evaluation and optimization

## References

1. DeWitt & Gray (1992). Parallel database systems: the future of high performance database systems.
2. Marcus & Papaemmanouil (2018). Deep reinforcement learning for join order enumeration.
3. Kraska et al. (2018). The case for learned index structures.
4. Ding et al. (2019). AI meets AI: Leveraging query executions to improve index recommendations.

## Authors

- Pravallika Nakarikanti (pn4154@g.rit.edu)
- Kolbe Yang (kky2806@rit.edu)
- Atharva Dhupkar (ad6258@rit.edu)

Rochester Institute of Technology, CSCI 725: Advanced Databases

## License

MIT License - See LICENSE file for details.
