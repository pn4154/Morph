#!/usr/bin/env python3
"""
Morph Evaluation Module

Provides baseline comparisons and comprehensive evaluation metrics
for the RL-based partition optimization system.
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import psycopg2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WorkloadMetrics:
    """Metrics from running a workload"""
    total_queries: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    std_latency_ms: float
    throughput_qps: float
    total_time_s: float
    failed_queries: int
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PartitioningMetrics:
    """Metrics about partition/shard state"""
    total_shards: int
    shards_per_node: Dict[str, int]
    size_per_node: Dict[str, int]
    balance_score: float  # 0-1, higher is more balanced
    cross_shard_query_ratio: float
    
    def to_dict(self) -> dict:
        return asdict(self)


class BaselineStrategy:
    """Base class for partitioning strategies"""
    
    def __init__(self, connection):
        self.conn = connection
    
    def apply(self) -> bool:
        """Apply the partitioning strategy"""
        raise NotImplementedError
    
    def name(self) -> str:
        raise NotImplementedError


class RoundRobinBaseline(BaselineStrategy):
    """Round-robin shard distribution baseline"""
    
    def name(self) -> str:
        return "round_robin"
    
    def apply(self) -> bool:
        """Rebalance shards using round-robin distribution"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT rebalance_table_shards()")
            return True
        except Exception as e:
            logger.error(f"Round-robin baseline failed: {e}")
            return False


class HashBaseline(BaselineStrategy):
    """Hash-based partitioning baseline (Citus default)"""
    
    def name(self) -> str:
        return "hash_default"
    
    def apply(self) -> bool:
        """Default Citus hash distribution - no action needed"""
        return True


class RangeBaseline(BaselineStrategy):
    """Range-based partitioning baseline"""
    
    def name(self) -> str:
        return "range_based"
    
    def apply(self) -> bool:
        """
        Note: Citus primarily uses hash distribution.
        Range partitioning would require recreating tables with
        time-based or value-based partitioning.
        """
        logger.info("Range-based partitioning requires schema changes")
        return True


class WorkloadAnalysisBaseline(BaselineStrategy):
    """Workload-aware baseline that analyzes query patterns"""
    
    def name(self) -> str:
        return "workload_aware"
    
    def apply(self) -> bool:
        """
        Analyze recent query patterns and adjust shard placement.
        This is a simplified version - a full implementation would
        parse query logs and optimize for colocation.
        """
        try:
            with self.conn.cursor() as cur:
                # Get query statistics
                cur.execute("""
                    SELECT query, calls, total_exec_time
                    FROM pg_stat_statements
                    ORDER BY total_exec_time DESC
                    LIMIT 100
                """)
                top_queries = cur.fetchall()
                
                # Analyze for table access patterns (simplified)
                table_access = {}
                for query, calls, time in top_queries:
                    for table in ['lineitem', 'orders', 'customer', 'part', 'supplier']:
                        if table in query.lower():
                            table_access[table] = table_access.get(table, 0) + calls
                
                logger.info(f"Table access patterns: {table_access}")
                
                # Could implement colocation optimization here
                return True
        except Exception as e:
            logger.error(f"Workload analysis failed: {e}")
            return False


class Evaluator:
    """Comprehensive evaluation of partitioning strategies"""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5432,
        database: str = 'morphdb',
        user: str = 'morph',
        password: str = 'MorphSecurePass123!'
    ):
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        self.conn.autocommit = True
    
    def get_tpch_queries(self) -> List[Dict[str, Any]]:
        """Get TPC-H query templates"""
        return [
            {
                "name": "Q1_pricing_summary",
                "type": "olap",
                "query": """
                    SELECT l_returnflag, l_linestatus,
                           SUM(l_quantity) as sum_qty,
                           SUM(l_extendedprice) as sum_base_price,
                           SUM(l_extendedprice * (1 - l_discount)) as sum_disc_price,
                           AVG(l_quantity) as avg_qty,
                           AVG(l_extendedprice) as avg_price,
                           COUNT(*) as count_order
                    FROM lineitem
                    WHERE l_shipdate <= DATE '1998-09-01'
                    GROUP BY l_returnflag, l_linestatus
                    ORDER BY l_returnflag, l_linestatus
                """
            },
            {
                "name": "Q3_shipping_priority",
                "type": "olap",
                "query": """
                    SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) as revenue,
                           o_orderdate, o_shippriority
                    FROM customer, orders, lineitem
                    WHERE c_mktsegment = 'BUILDING'
                      AND c_custkey = o_custkey
                      AND l_orderkey = o_orderkey
                      AND o_orderdate < DATE '1995-03-15'
                      AND l_shipdate > DATE '1995-03-15'
                    GROUP BY l_orderkey, o_orderdate, o_shippriority
                    ORDER BY revenue DESC, o_orderdate
                    LIMIT 10
                """
            },
            {
                "name": "Q6_forecasting",
                "type": "olap",
                "query": """
                    SELECT SUM(l_extendedprice * l_discount) as revenue
                    FROM lineitem
                    WHERE l_shipdate >= DATE '1994-01-01'
                      AND l_shipdate < DATE '1995-01-01'
                      AND l_discount BETWEEN 0.05 AND 0.07
                      AND l_quantity < 24
                """
            },
            {
                "name": "point_lookup_customer",
                "type": "oltp",
                "query": "SELECT * FROM customer WHERE c_custkey = %s",
                "params_fn": lambda: (np.random.randint(1, 150000),)
            },
            {
                "name": "point_lookup_order",
                "type": "oltp",
                "query": "SELECT * FROM orders WHERE o_orderkey = %s",
                "params_fn": lambda: (np.random.randint(1, 1500000),)
            },
            {
                "name": "range_scan_lineitem",
                "type": "olap",
                "query": """
                    SELECT COUNT(*), SUM(l_quantity), AVG(l_extendedprice)
                    FROM lineitem
                    WHERE l_orderkey BETWEEN %s AND %s
                """,
                "params_fn": lambda: (np.random.randint(1, 1400000), 
                                     np.random.randint(1400000, 1500000))
            }
        ]
    
    def run_workload(
        self,
        workload_type: str = 'mixed',
        duration_s: int = 60,
        target_qps: Optional[float] = None
    ) -> WorkloadMetrics:
        """Run a workload and collect metrics"""
        
        queries = self.get_tpch_queries()
        
        if workload_type == 'oltp':
            queries = [q for q in queries if q['type'] == 'oltp']
        elif workload_type == 'olap':
            queries = [q for q in queries if q['type'] == 'olap']
        
        latencies = []
        failed = 0
        start_time = time.time()
        query_count = 0
        
        with self.conn.cursor() as cur:
            while time.time() - start_time < duration_s:
                query_info = np.random.choice(queries)
                query = query_info['query']
                params = query_info.get('params_fn', lambda: None)()
                
                query_start = time.perf_counter()
                try:
                    cur.execute(query, params)
                    cur.fetchall()
                    latency_ms = (time.perf_counter() - query_start) * 1000
                    latencies.append(latency_ms)
                except Exception as e:
                    logger.warning(f"Query failed: {e}")
                    failed += 1
                
                query_count += 1
                
                # Rate limiting if target_qps specified
                if target_qps:
                    expected_time = query_count / target_qps
                    elapsed = time.time() - start_time
                    if elapsed < expected_time:
                        time.sleep(expected_time - elapsed)
        
        total_time = time.time() - start_time
        
        if not latencies:
            latencies = [0]
        
        return WorkloadMetrics(
            total_queries=len(latencies),
            mean_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            max_latency_ms=float(np.max(latencies)),
            min_latency_ms=float(np.min(latencies)),
            std_latency_ms=float(np.std(latencies)),
            throughput_qps=len(latencies) / total_time,
            total_time_s=total_time,
            failed_queries=failed
        )
    
    def get_partition_metrics(self) -> PartitioningMetrics:
        """Get current partition/shard metrics"""
        with self.conn.cursor() as cur:
            # Get shard distribution
            cur.execute("""
                SELECT nodename, COUNT(*) as shard_count
                FROM pg_dist_shard_placement
                GROUP BY nodename
                ORDER BY nodename
            """)
            shards_per_node = {row[0]: row[1] for row in cur.fetchall()}
            
            total_shards = sum(shards_per_node.values())
            
            # Calculate balance score
            if total_shards > 0:
                counts = list(shards_per_node.values())
                mean_count = np.mean(counts)
                std_count = np.std(counts)
                balance_score = 1.0 - (std_count / (mean_count + 1))
            else:
                balance_score = 1.0
            
            # Get size per node (simplified - would need shard sizes)
            size_per_node = {node: count * 1000000 for node, count in shards_per_node.items()}
            
            return PartitioningMetrics(
                total_shards=total_shards,
                shards_per_node=shards_per_node,
                size_per_node=size_per_node,
                balance_score=float(balance_score),
                cross_shard_query_ratio=0.0  # Would need query analysis
            )
    
    def evaluate_baseline(
        self,
        baseline: BaselineStrategy,
        workload_types: List[str] = ['oltp', 'olap', 'mixed'],
        duration_per_workload_s: int = 60
    ) -> Dict[str, Any]:
        """Evaluate a baseline strategy"""
        
        logger.info(f"Evaluating baseline: {baseline.name()}")
        
        # Apply baseline strategy
        success = baseline.apply()
        if not success:
            logger.error(f"Failed to apply baseline: {baseline.name()}")
        
        results = {
            "baseline": baseline.name(),
            "timestamp": datetime.now().isoformat(),
            "workloads": {}
        }
        
        # Get initial partition metrics
        results["partition_metrics"] = self.get_partition_metrics().to_dict()
        
        # Run each workload type
        for workload_type in workload_types:
            logger.info(f"  Running {workload_type} workload for {duration_per_workload_s}s...")
            metrics = self.run_workload(
                workload_type=workload_type,
                duration_s=duration_per_workload_s
            )
            results["workloads"][workload_type] = metrics.to_dict()
            logger.info(f"    Mean latency: {metrics.mean_latency_ms:.2f}ms, "
                       f"Throughput: {metrics.throughput_qps:.1f} QPS")
        
        return results
    
    def run_full_evaluation(
        self,
        output_dir: str = './evaluation_results',
        duration_per_workload_s: int = 60
    ) -> Dict[str, Any]:
        """Run full evaluation comparing all baselines"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        baselines = [
            HashBaseline(self.conn),
            RoundRobinBaseline(self.conn),
            WorkloadAnalysisBaseline(self.conn)
        ]
        
        all_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "duration_per_workload_s": duration_per_workload_s,
            "baselines": {}
        }
        
        for baseline in baselines:
            results = self.evaluate_baseline(
                baseline,
                duration_per_workload_s=duration_per_workload_s
            )
            all_results["baselines"][baseline.name()] = results
        
        # Save results
        results_file = output_path / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Generate summary
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for baseline_name, baseline_results in results["baselines"].items():
            print(f"\n{baseline_name}:")
            print("-"*40)
            
            for workload, metrics in baseline_results["workloads"].items():
                print(f"  {workload}:")
                print(f"    Mean Latency: {metrics['mean_latency_ms']:.2f}ms")
                print(f"    P95 Latency:  {metrics['p95_latency_ms']:.2f}ms")
                print(f"    Throughput:   {metrics['throughput_qps']:.1f} QPS")
        
        print("\n" + "="*60)
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def compare_with_rl_agent(
    evaluator: Evaluator,
    agent_results_path: str,
    output_dir: str = './comparison_results'
) -> Dict[str, Any]:
    """Compare RL agent results with baselines"""
    
    # Load RL agent results
    with open(agent_results_path, 'r') as f:
        agent_results = json.load(f)
    
    # Run baseline evaluations
    baseline_results = evaluator.run_full_evaluation(
        output_dir=output_dir,
        duration_per_workload_s=60
    )
    
    # Compare
    comparison = {
        "rl_agent": agent_results,
        "baselines": baseline_results["baselines"],
        "improvements": {}
    }
    
    # Calculate improvements
    if "mean_reward" in agent_results:
        # Extract comparable metrics
        for baseline_name, baseline_data in baseline_results["baselines"].items():
            if "mixed" in baseline_data["workloads"]:
                baseline_latency = baseline_data["workloads"]["mixed"]["mean_latency_ms"]
                comparison["improvements"][baseline_name] = {
                    "baseline_latency_ms": baseline_latency,
                    # RL comparison would go here
                }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison_file = output_path / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Evaluate partitioning strategies')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5432)
    parser.add_argument('--database', type=str, default='morphdb')
    parser.add_argument('--user', type=str, default='morph')
    parser.add_argument('--password', type=str, default='MorphSecurePass123!')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration per workload in seconds')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results')
    parser.add_argument('--compare-rl', type=str, default=None,
                       help='Path to RL agent results for comparison')
    
    args = parser.parse_args()
    
    evaluator = Evaluator(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password
    )
    
    try:
        if args.compare_rl:
            results = compare_with_rl_agent(
                evaluator,
                args.compare_rl,
                args.output_dir
            )
        else:
            results = evaluator.run_full_evaluation(
                output_dir=args.output_dir,
                duration_per_workload_s=args.duration
            )
        
        print("\nEvaluation complete!")
    finally:
        evaluator.close()


if __name__ == '__main__':
    main()
