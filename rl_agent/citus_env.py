"""
Morph: Citus Database Environment for Reinforcement Learning
This environment interfaces with a distributed Citus PostgreSQL cluster
and exposes partitioning decisions as an RL action space.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import psycopg2
from psycopg2 import sql
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionType(IntEnum):
    """Discrete action types for partition management"""
    NO_OP = 0
    MOVE_PARTITION = 1
    SPLIT_PARTITION = 2
    MERGE_PARTITIONS = 3
    REBALANCE_PARTITION = 4
    MOVE_DATA_RANGE = 5


@dataclass
class ShardInfo:
    """Information about a shard in the Citus cluster"""
    shard_id: int
    table_name: str
    node_name: str
    node_port: int
    shard_size: int
    min_value: Optional[int]
    max_value: Optional[int]


@dataclass
class NodeInfo:
    """Information about a worker node"""
    node_name: str
    node_port: int
    shard_count: int
    total_size: int
    cpu_usage: float
    memory_usage: float
    active_connections: int


class CitusConnection:
    """Manages connections to the Citus coordinator and workers"""
    
    def __init__(
        self,
        coordinator_host: str = "localhost",
        coordinator_port: int = 5432,
        database: str = "morphdb",
        user: str = "morph",
        password: str = "MorphSecurePass123!"
    ):
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.database = database
        self.user = user
        self.password = password
        self._conn = None
    
    def connect(self) -> psycopg2.extensions.connection:
        """Establish connection to coordinator"""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=self.coordinator_host,
                port=self.coordinator_port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self._conn.autocommit = True
        return self._conn
    
    def execute(self, query: str, params: tuple = None) -> List[tuple]:
        """Execute a query and return results"""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(query, params)
            if cur.description:
                return cur.fetchall()
            return []
    
    def close(self):
        """Close the connection"""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None


class CitusPartitionManager:
    """Manages partition operations in the Citus cluster"""
    
    def __init__(self, connection: CitusConnection):
        self.conn = connection
    
    def get_worker_nodes(self) -> List[NodeInfo]:
        """Get information about all worker nodes"""
        query = """
        SELECT 
            nodename,
            nodeport
        FROM pg_dist_node
        WHERE noderole = 'primary' AND nodename LIKE '%worker%'
        ORDER BY nodename
        """
        try:
            results = self.conn.execute(query)
            nodes = []
            for row in results:
                # Get shard count for this node
                shard_query = """
                SELECT COUNT(*) FROM pg_dist_shard_placement WHERE nodename = %s
                """
                shard_result = self.conn.execute(shard_query, (row[0],))
                shard_count = shard_result[0][0] if shard_result else 0
                
                nodes.append(NodeInfo(
                    node_name=row[0],
                    node_port=row[1],
                    shard_count=shard_count,
                    total_size=0,
                    cpu_usage=0.0,
                    memory_usage=0.0,
                    active_connections=0
                ))
            return nodes
        except Exception as e:
            logger.warning(f"Error getting worker nodes: {e}")
            return []
    
    def get_shard_placements(self) -> List[ShardInfo]:
        """Get all shard placements in the cluster - simplified query without size calculation"""
        query = """
        SELECT 
            s.shardid,
            s.logicalrelid::text as table_name,
            sp.nodename,
            sp.nodeport,
            s.shardminvalue,
            s.shardmaxvalue
        FROM pg_dist_shard s
        JOIN pg_dist_shard_placement sp ON s.shardid = sp.shardid
        WHERE sp.nodename LIKE '%worker%'
        ORDER BY s.logicalrelid, s.shardminvalue
        """
        try:
            results = self.conn.execute(query)
            shards = []
            for row in results:
                min_val = None
                max_val = None
                try:
                    if row[4] is not None:
                        min_val = int(row[4])
                    if row[5] is not None:
                        max_val = int(row[5])
                except (ValueError, TypeError):
                    pass
                
                shards.append(ShardInfo(
                    shard_id=row[0],
                    table_name=row[1],
                    node_name=row[2],
                    node_port=row[3],
                    shard_size=1000000,  # Default size estimate
                    min_value=min_val,
                    max_value=max_val
                ))
            return shards
        except Exception as e:
            logger.warning(f"Error getting shard placements: {e}")
            return []
    
    def move_shard(self, shard_id: int, target_node: str, target_port: int = 5432) -> bool:
        """Move a shard to a different worker node"""
        try:
            # Get current placement
            query = """
            SELECT nodename, nodeport 
            FROM pg_dist_shard_placement 
            WHERE shardid = %s
            """
            result = self.conn.execute(query, (shard_id,))
            if not result:
                logger.error(f"Shard {shard_id} not found")
                return False
            
            source_node, source_port = result[0]
            
            if source_node == target_node:
                logger.info(f"Shard {shard_id} already on {target_node}")
                return True
            
            # Use Citus shard movement
            move_query = """
            SELECT citus_move_shard_placement(%s, %s, %s, %s, %s)
            """
            self.conn.execute(move_query, (
                shard_id, source_node, source_port, target_node, target_port
            ))
            logger.info(f"Moved shard {shard_id} from {source_node} to {target_node}")
            return True
        except Exception as e:
            logger.error(f"Error moving shard {shard_id}: {e}")
            return False
    
    def rebalance_shards(self) -> bool:
        """Rebalance all shards across worker nodes"""
        try:
            self.conn.execute("SELECT rebalance_table_shards()")
            logger.info("Shard rebalancing completed")
            return True
        except Exception as e:
            logger.error(f"Error rebalancing shards: {e}")
            return False
    
    def get_shard_distribution(self) -> Dict[str, int]:
        """Get shard count per node"""
        query = """
        SELECT nodename, COUNT(*) as shard_count
        FROM pg_dist_shard_placement
        WHERE nodename LIKE '%worker%'
        GROUP BY nodename
        ORDER BY nodename
        """
        try:
            results = self.conn.execute(query)
            return {row[0]: row[1] for row in results}
        except Exception as e:
            logger.warning(f"Error getting shard distribution: {e}")
            return {}


class WorkloadGenerator:
    """Generates and executes TPC-H style workloads"""
    
    def __init__(self, connection: CitusConnection):
        self.conn = connection
        self.query_templates = self._load_query_templates()
    
    def _load_query_templates(self) -> Dict[str, str]:
        """Load TPC-H query templates"""
        return {
            "q1_pricing_summary": """
                SELECT 
                    l_returnflag,
                    l_linestatus,
                    SUM(l_quantity) as sum_qty,
                    SUM(l_extendedprice) as sum_base_price,
                    AVG(l_quantity) as avg_qty,
                    COUNT(*) as count_order
                FROM lineitem
                WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '%s days'
                GROUP BY l_returnflag, l_linestatus
                ORDER BY l_returnflag, l_linestatus
            """,
            "q6_forecasting_revenue": """
                SELECT SUM(l_extendedprice * l_discount) as revenue
                FROM lineitem
                WHERE l_shipdate >= DATE %s
                  AND l_shipdate < DATE %s + INTERVAL '1 year'
                  AND l_discount BETWEEN %s - 0.01 AND %s + 0.01
                  AND l_quantity < %s
            """,
            "point_lookup_customer": """
                SELECT * FROM customer WHERE c_custkey = %s
            """,
            "point_lookup_order": """
                SELECT * FROM orders WHERE o_orderkey = %s
            """,
            "range_scan_lineitem": """
                SELECT COUNT(*), SUM(l_quantity), AVG(l_extendedprice)
                FROM lineitem
                WHERE l_orderkey BETWEEN %s AND %s
            """
        }
    
    def execute_query(self, query_name: str, params: tuple = None) -> Tuple[float, int]:
        """Execute a query and return (latency_ms, rows_affected)"""
        if query_name not in self.query_templates:
            raise ValueError(f"Unknown query: {query_name}")
        
        query = self.query_templates[query_name]
        
        start_time = time.perf_counter()
        try:
            results = self.conn.execute(query, params)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return latency_ms, len(results)
        except Exception as e:
            logger.error(f"Query {query_name} failed: {e}")
            return 10000.0, 0
    
    def run_workload_batch(
        self,
        batch_size: int = 100,
        workload_type: str = "mixed"
    ) -> Dict[str, Any]:
        """Run a batch of queries and return performance metrics"""
        latencies = []
        query_counts = {name: 0 for name in self.query_templates}
        
        for _ in range(batch_size):
            query_name, params = self._sample_query(workload_type)
            latency, _ = self.execute_query(query_name, params)
            latencies.append(latency)
            query_counts[query_name] += 1
        
        return {
            "total_queries": batch_size,
            "mean_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "max_latency_ms": np.max(latencies),
            "query_distribution": query_counts
        }
    
    def _sample_query(self, workload_type: str) -> Tuple[str, tuple]:
        """Sample a query based on workload type"""
        if workload_type == "oltp":
            query_name = np.random.choice([
                "point_lookup_customer",
                "point_lookup_order"
            ], p=[0.5, 0.5])
            if query_name == "point_lookup_customer":
                params = (np.random.randint(1, 15000),)
            else:
                params = (np.random.randint(1, 150000),)
        
        elif workload_type == "olap":
            query_name = np.random.choice([
                "q1_pricing_summary",
                "q6_forecasting_revenue",
                "range_scan_lineitem"
            ])
            if query_name == "q1_pricing_summary":
                params = (np.random.randint(60, 120),)
            elif query_name == "q6_forecasting_revenue":
                year = np.random.choice(["1993-01-01", "1994-01-01", "1995-01-01"])
                discount = 0.05 + np.random.random() * 0.04
                params = (year, year, discount, discount, 24 + np.random.randint(0, 2))
            else:
                start_key = np.random.randint(1, 140000)
                params = (start_key, start_key + 10000)
        
        else:  # mixed
            if np.random.random() < 0.7:
                return self._sample_query("oltp")
            else:
                return self._sample_query("olap")
        
        return query_name, params


class MorphEnv(gym.Env):
    """
    Gymnasium environment for the Morph RL agent.
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        coordinator_host: str = "localhost",
        coordinator_port: int = 5432,
        database: str = "morphdb",
        user: str = "morph",
        password: str = "MorphSecurePass123!",
        num_workers: int = 3,
        max_shards_per_table: int = 32,
        episode_length: int = 100,
        workload_type: str = "mixed",
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.num_workers = num_workers
        self.max_shards = max_shards_per_table
        self.episode_length = episode_length
        self.workload_type = workload_type
        self.render_mode = render_mode
        
        # Database connections
        self.connection = CitusConnection(
            coordinator_host=coordinator_host,
            coordinator_port=coordinator_port,
            database=database,
            user=user,
            password=password
        )
        self.partition_manager = CitusPartitionManager(self.connection)
        self.workload_generator = WorkloadGenerator(self.connection)
        
        # State tracking
        self.current_step = 0
        self.episode_rewards = []
        self.baseline_latency = None
        
        # Define observation space
        obs_dim = num_workers * 2 + 8
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Define action space (Hybrid: Discrete + Continuous)
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(len(ActionType)),
            "parameters": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32
            )
        })
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment for a new episode"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_rewards = []
        
        # Get baseline performance
        try:
            baseline_metrics = self.workload_generator.run_workload_batch(
                batch_size=50,
                workload_type=self.workload_type
            )
            self.baseline_latency = baseline_metrics["mean_latency_ms"]
        except Exception as e:
            logger.warning(f"Error getting baseline: {e}")
            self.baseline_latency = 100.0
        
        obs = self._get_observation()
        info = {
            "baseline_latency": self.baseline_latency,
            "shard_distribution": self.partition_manager.get_shard_distribution()
        }
        
        return obs, info
    
    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute an action and return the new state"""
        self.current_step += 1
        
        action_type = ActionType(action["action_type"])
        params = action["parameters"]
        
        # Execute the action
        action_success, action_cost = self._execute_action(action_type, params)
        
        # Run workload to measure performance
        try:
            metrics = self.workload_generator.run_workload_batch(
                batch_size=100,
                workload_type=self.workload_type
            )
        except Exception as e:
            logger.warning(f"Error running workload: {e}")
            metrics = {"mean_latency_ms": 1000.0, "p95_latency_ms": 2000.0}
        
        # Calculate reward
        reward = self._calculate_reward(metrics, action_cost, action_success)
        self.episode_rewards.append(reward)
        
        # Get new observation
        obs = self._get_observation()
        
        # Check termination
        terminated = False
        truncated = self.current_step >= self.episode_length
        
        info = {
            "action_type": action_type.name,
            "action_success": action_success,
            "mean_latency_ms": metrics["mean_latency_ms"],
            "p95_latency_ms": metrics.get("p95_latency_ms", 0),
            "shard_distribution": self.partition_manager.get_shard_distribution(),
            "cumulative_reward": sum(self.episode_rewards)
        }
        
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector"""
        # Get shard distribution
        shard_dist = self.partition_manager.get_shard_distribution()
        total_shards = sum(shard_dist.values()) if shard_dist else 1
        
        # Normalize shard counts per node
        shard_counts = np.zeros(self.num_workers)
        for i, (node, count) in enumerate(sorted(shard_dist.items())):
            if i < self.num_workers:
                shard_counts[i] = count / max(total_shards, 1)
        
        # Get node sizes (placeholder)
        node_sizes = np.random.uniform(0.2, 0.8, self.num_workers)
        
        # Latency statistics from recent workload
        try:
            metrics = self.workload_generator.run_workload_batch(
                batch_size=20,
                workload_type=self.workload_type
            )
            latency_stats = np.array([
                metrics["mean_latency_ms"] / 1000,
                metrics["p50_latency_ms"] / 1000,
                metrics["p95_latency_ms"] / 1000,
                metrics["p99_latency_ms"] / 1000
            ])
        except Exception:
            latency_stats = np.array([0.1, 0.1, 0.2, 0.3])
        
        # Workload features
        workload_features = np.array([
            0.7 if self.workload_type == "oltp" else 0.3,
            0.3 if self.workload_type == "oltp" else 0.7,
            np.random.uniform(0.3, 0.7),
            self.current_step / self.episode_length
        ])
        
        obs = np.concatenate([
            shard_counts,
            node_sizes,
            np.clip(latency_stats, 0, 1),
            workload_features
        ]).astype(np.float32)
        
        return obs
    
    def _execute_action(
        self,
        action_type: ActionType,
        params: np.ndarray
    ) -> Tuple[bool, float]:
        """Execute the specified action"""
        if action_type == ActionType.NO_OP:
            return True, 0.0
        
        elif action_type == ActionType.MOVE_PARTITION:
            shards = self.partition_manager.get_shard_placements()
            if not shards:
                return False, 0.0
            
            shard_idx = int(params[0] * len(shards))
            shard_idx = min(shard_idx, len(shards) - 1)
            shard = shards[shard_idx]
            
            nodes = self.partition_manager.get_worker_nodes()
            if not nodes:
                return False, 0.0
            
            target_idx = int(params[1] * len(nodes))
            target_idx = min(target_idx, len(nodes) - 1)
            target_node = nodes[target_idx]
            
            success = self.partition_manager.move_shard(
                shard.shard_id,
                target_node.node_name,
                target_node.node_port
            )
            
            cost = 0.5 if success else 0.0
            return success, cost
        
        elif action_type == ActionType.REBALANCE_PARTITION:
            success = self.partition_manager.rebalance_shards()
            return success, 1.0
        
        elif action_type == ActionType.SPLIT_PARTITION:
            logger.info("SPLIT_PARTITION requires custom implementation")
            return False, 0.0
        
        elif action_type == ActionType.MERGE_PARTITIONS:
            logger.info("MERGE_PARTITIONS requires custom implementation")
            return False, 0.0
        
        elif action_type == ActionType.MOVE_DATA_RANGE:
            logger.info("MOVE_DATA_RANGE requires custom implementation")
            return False, 0.0
        
        return False, 0.0
    
    def _calculate_reward(
        self,
        metrics: Dict[str, Any],
        action_cost: float,
        action_success: bool
    ) -> float:
        """Calculate reward based on performance metrics"""
        # Latency reward
        if self.baseline_latency and self.baseline_latency > 0:
            latency_ratio = self.baseline_latency / max(metrics["mean_latency_ms"], 1)
            latency_reward = np.clip(latency_ratio - 1.0, -1.0, 1.0)
        else:
            latency_reward = -metrics["mean_latency_ms"] / 1000
        
        # Load balance reward
        shard_dist = self.partition_manager.get_shard_distribution()
        if shard_dist:
            counts = list(shard_dist.values())
            if len(counts) > 0 and np.mean(counts) > 0:
                balance_score = 1.0 - (np.std(counts) / (np.mean(counts) + 1))
                balance_reward = balance_score * 0.2
            else:
                balance_reward = 0.0
        else:
            balance_reward = 0.0
        
        cost_penalty = -action_cost * 0.1
        failure_penalty = 0.0 if action_success else -0.5
        
        total_reward = latency_reward + balance_reward + cost_penalty + failure_penalty
        
        return float(total_reward)
    
    def render(self):
        """Render the current state"""
        if self.render_mode == "ansi":
            shard_dist = self.partition_manager.get_shard_distribution()
            output = f"\n=== Step {self.current_step} ===\n"
            output += f"Shard Distribution: {shard_dist}\n"
            if self.episode_rewards:
                output += f"Last Reward: {self.episode_rewards[-1]:.4f}\n"
                output += f"Cumulative: {sum(self.episode_rewards):.4f}\n"
            return output
        elif self.render_mode == "human":
            print(self.render())
    
    def close(self):
        """Clean up resources"""
        self.connection.close()


class MorphEnvFlat(MorphEnv):
    """Flattened action space version for PPO compatibility."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32
        )
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Convert flat action to dict and execute"""
        action_logits = action[:6]
        action_type = int(np.argmax(action_logits))
        params = (action[6:10] + 1) / 2
        
        dict_action = {
            "action_type": action_type,
            "parameters": params
        }
        
        return super().step(dict_action)


def make_env(env_id: str = "morph", **kwargs) -> gym.Env:
    """Factory function to create the environment"""
    if env_id == "morph":
        return MorphEnv(**kwargs)
    elif env_id == "morph-flat":
        return MorphEnvFlat(**kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_id}")


if __name__ == "__main__":
    env = MorphEnvFlat(
        coordinator_host="localhost",
        coordinator_port=5555,
        render_mode="ansi"
    )
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, action_type={info['action_type']}")
    
    env.close()
