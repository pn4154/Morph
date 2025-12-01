import time
import json
import os
import sys
from datetime import datetime
from typing import Dict, List
import numpy as np
import psycopg2
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "workload"))
sys.path.insert(0, str(project_root / "evaluation"))

from generator import WorkloadGenerator
from access_monitoring import AccessPatternMonitor
from baseline_partitioning import BaselinePartitioner

class BenchmarkRunner:
    def __init__(self, db_config, output_dir='./benchmark_results'):
        self.db_config = db_config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def run_baseline_evaluation(self, 
                                strategies=['hash', 'range'],
                                workload_types=['static', 'zipfian'],
                                num_queries=1000,
                                num_trials=10):
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'db_config': {k: v for k, v in self.db_config.items() if k != 'password'},
                'num_queries': num_queries,
                'num_trials': num_trials
            },
            'experiments': []
        }
        
        for strategy in strategies:
            for workload_type in workload_types:
                print(f"\n{'='*60}")
                print(f"Testing: {strategy} partitioning with {workload_type} workload")
                print(f"{'='*60}")
                
                experiment_results = self._run_experiment(
                    strategy=strategy,
                    workload_type=workload_type,
                    num_queries=num_queries,
                    num_trials=num_trials
                )
                
                results['experiments'].append(experiment_results)

                self._save_results(results)
        
        print(f"Results saved to: {self.output_dir}")
        
        return results
    
    def _run_experiment(self, strategy, workload_type, num_queries, num_trials):
        experiment = {
            'strategy': strategy,
            'workload_type': workload_type,
            'trials': []
        }
        
        for trial in range(num_trials):
            # print(f"Trial {trial + 1}/{num_trials}")
            
            self._setup_partition_strategy(strategy)
            
            trial_results = self._run_trial(workload_type, num_queries)
            
            trial_metrics = self._collect_metrics(trial_results)
            
            experiment['trials'].append(trial_metrics)
            
            # print(f"Median latency: {trial_metrics['latency_p50']:.4f}s")
            # print(f"P95 latency: {trial_metrics['latency_p95']:.4f}s")
        
        # Calculate aggregate statistics
        experiment['summary'] = self._calculate_summary_stats(experiment['trials'])
        
        return experiment
    
    def _setup_partition_strategy(self, strategy):
        partitioner = BaselinePartitioner(self.db_config)
        
        try:
            partitioner.reset_table('orders')
            partitioner.reset_table('customer')
            
            if strategy == 'hash':
                partitioner.hash_partition('orders', 'o_custkey')
                partitioner.hash_partition('customer', 'c_custkey')
            elif strategy == 'range':
                partitioner.range_partition('orders', 'o_custkey')
                partitioner.range_partition('customer', 'c_custkey')
            elif strategy == 'round_robin':
                partitioner.round_robin_partition('orders')
            
            print(f"    {strategy} partitioning applied")
            
        finally:
            partitioner.close()
    
    def _run_trial(self, workload_type, num_queries):
        generator = WorkloadGenerator(self.db_config)
        monitor = AccessPatternMonitor(self.db_config)
        
        generator.run_workload(num_queries=50, distribution='uniform')
        
        results = generator.run_workload(
            num_queries=num_queries,
            distribution=workload_type
        )
        for result in results:
            if result['success']:
                monitor.track_query(
                    query_text=result['query'],
                    params=result['params'],
                    execution_time=result['execution_time'],
                    rows_returned=result['rows_returned']
                )
        
        generator.close()
        
        return {
            'workload_results': results,
            'access_patterns': {
                'frequency': monitor.get_access_heatmap(),
                'co_access': monitor.get_co_access_patterns()
            }
        }
    
    def _collect_metrics(self, trial_results):
        workload_results = trial_results['workload_results']
        
        latencies = [r['execution_time'] for r in workload_results if r['success']]
        
        metrics = {
            'latency_p50': np.percentile(latencies, 50),
            'latency_p95': np.percentile(latencies, 95),
            'latency_p99': np.percentile(latencies, 99),
            'latency_mean': np.mean(latencies),
            'latency_std': np.std(latencies),
            'throughput': len(latencies) / sum(latencies) if latencies else 0,
            'success_rate': sum(1 for r in workload_results if r['success']) / len(workload_results),
            'total_queries': len(workload_results)
        }
        
        metrics.update(self._analyze_cross_partition_queries())
        
        return metrics
    
    def _analyze_cross_partition_queries(self):
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(mean_exec_time) as avg_exec_time
                FROM citus_stat_statements
                WHERE calls > 0
                LIMIT 1;
            """)
            
            row = cursor.fetchone()
            
            if row and row[0] > 0:
                return {
                    'cross_partition_ratio': 0.0,
                    'avg_distributed_query_time': row[1] or 0,
                    'avg_local_query_time': row[1] or 0
                }
            
        except Exception as e:
            pass
        
        finally:
            cursor.close()
            conn.close()
        
        return {
            'cross_partition_ratio': 0,
            'avg_distributed_query_time': 0,
            'avg_local_query_time': 0
        }
    
    def _calculate_summary_stats(self, trials):
        metrics_to_aggregate = [
            'latency_p50', 'latency_p95', 'latency_p99',
            'latency_mean', 'throughput', 'cross_partition_ratio'
        ]
        
        summary = {}
        
        for metric in metrics_to_aggregate:
            values = [trial[metric] for trial in trials if metric in trial]
            if values:
                summary[f'{metric}_median'] = np.median(values)
                summary[f'{metric}_iqr'] = np.percentile(values, 75) - np.percentile(values, 25)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
        
        return summary
    
    def _save_results(self, results):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_dir, f'baseline_results_{timestamp}.json')
    
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
    
        results = convert_numpy(results)
    
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
    
        print(f"\n  Results saved to: {filename}")
    
    def generate_comparison_report(self, results_file=None):
        """Generate a comparison report from results"""
        
        if results_file is None:
            files = [f for f in os.listdir(self.output_dir) if f.startswith('baseline_results_')]
            if not files:
                print("No results files found")
                return
            results_file = os.path.join(self.output_dir, sorted(files)[-1])
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Save report to file
        report_file = results_file.replace('.json', '_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BASELINE PARTITIONING COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            for exp in results['experiments']:
                strategy = exp['strategy']
                workload = exp['workload_type']
                summary = exp['summary']
                
                f.write(f"{strategy.upper()} Partitioning - {workload.capitalize()} Workload\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Median P50 Latency: {summary['latency_p50_median']*1000:.2f} ms "
                       f"(IQR: {summary['latency_p50_iqr']*1000:.2f} ms)\n")
                f.write(f"  Median P95 Latency: {summary['latency_p95_median']*1000:.2f} ms "
                       f"(IQR: {summary['latency_p95_iqr']*1000:.2f} ms)\n")
                f.write(f"  Median Throughput: {summary['throughput_median']:.2f} queries/sec\n")
                if 'cross_partition_ratio_median' in summary:
                    f.write(f"  Cross-partition Query Ratio: {summary['cross_partition_ratio_median']*100:.1f}%\n")
                f.write("\n")
        
        print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    DB_CONFIG = {
        'dbname': 'tpch_db',
        'user': 'partitionuser',
        'password': 'partitionpass',
        'host': 'localhost',
        'port': 5432
    }
    
    runner = BenchmarkRunner(DB_CONFIG)
    
    results = runner.run_baseline_evaluation(
        strategies=['hash', 'range'],
        workload_types=['uniform', 'zipfian'],
        num_queries=50,
        num_trials=3
    )

    runner.generate_comparison_report()