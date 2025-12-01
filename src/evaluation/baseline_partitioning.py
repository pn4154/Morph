import psycopg2
import hashlib
from typing import List, Tuple, Dict
import time

class BaselinePartitioner:
    def __init__(self, db_config, num_partitions=3):
        self.db_config = db_config
        self.num_partitions = num_partitions
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()
    
    def hash_partition(self, table='orders', key_column='o_custkey'):
        try:
            self.cursor.execute(f"""
                SELECT create_distributed_table('{table}', '{key_column}');
            """)
            self.conn.commit()
            
            self.cursor.execute(f"""
                SELECT nodename, nodeport, COUNT(*) as shard_count
                FROM citus_shards
                WHERE table_name::text = '{table}'
                GROUP BY nodename, nodeport;
            """)
            
            distribution = self.cursor.fetchall()
            return {'strategy': 'hash', 'distribution': distribution}
            
        except Exception as e:
            print(f"Error in hash partitioning: {e}")
            self.conn.rollback()
            return None
    
    def range_partition(self, table='orders', key_column='o_custkey'):
        try:
            self.cursor.execute(f"""
                SELECT MIN({key_column}), MAX({key_column})
                FROM {table};
            """)
            min_val, max_val = self.cursor.fetchone()
            
            range_size = (max_val - min_val) // self.num_partitions
            boundaries = [min_val + i * range_size for i in range(self.num_partitions + 1)]
            boundaries[-1] = max_val + 1 
            self.cursor.execute(f"""
                SELECT create_distributed_table('{table}', '{key_column}', 
                    colocate_with => 'none');
            """)
            self.conn.commit()
            
            return {
                'strategy': 'range',
                'boundaries': boundaries,
                'min': min_val,
                'max': max_val
            }
            
        except Exception as e:
            print(f"Error in range partitioning: {e}")
            self.conn.rollback()
            return None
    
    def round_robin_partition(self, table='orders'):
        try:
            # In Citus, approximate round-robin using hash distribution
            self.cursor.execute(f"""
                SELECT create_distributed_table('{table}', 'o_orderkey');
            """)
            self.conn.commit()
            
            return {'strategy': 'round_robin'}
            
        except Exception as e:
            print(f"Error in round-robin partitioning: {e}")
            self.conn.rollback()
            return None
    
    def get_partition_stats(self, table='orders'):
        self.cursor.execute(f"""
            SELECT 
                nodename,
                nodeport,
                shardid,
                shard_size / (1024.0 * 1024.0) as size_mb
            FROM citus_shards
            WHERE table_name::text = '{table}'
            ORDER BY nodename, shardid;
        """)
        
        stats = self.cursor.fetchall()
        sizes = [float(row[3]) for row in stats]  # Convert to float
        return {
            'total_shards': len(stats),
            'total_size_mb': sum(sizes),
            'avg_size_mb': sum(sizes) / len(sizes) if sizes else 0,
            'min_size_mb': min(sizes) if sizes else 0,
            'max_size_mb': max(sizes) if sizes else 0,
            'std_dev': self._std_dev(sizes),
            'details': stats
        }

    def _std_dev(self, values):
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def reset_table(self, table='orders'):
        try:
            self.cursor.execute(f"""
                SELECT undistribute_table('{table}');
            """)
            self.conn.commit()
        except Exception as e:
            print(f"Could not undistribute: {e}")
            self.conn.rollback()
    
    def close(self):
        self.cursor.close()
        self.conn.close()


if __name__ == "__main__":
    DB_CONFIG = {
        'dbname': 'tpch_db',
        'user': 'partitionuser',
        'password': 'partitionpass',
        'host': 'localhost',
        'port': 5432
    }
    
    partitioner = BaselinePartitioner(DB_CONFIG, num_partitions=3)
    
    result = partitioner.hash_partition('orders', 'o_custkey')
    print(f"\nPartition setup result: {result}")
    
    stats = partitioner.get_partition_stats('orders')
    print(f"\nPartition statistics:")
    print(f"  Total shards: {stats['total_shards']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")
    print(f"  Avg size: {stats['avg_size_mb']:.2f} MB")
    print(f"  Std dev: {stats['std_dev']:.2f} MB")
    
    partitioner.close()