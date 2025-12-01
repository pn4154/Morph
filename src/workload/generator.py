import random
import time
import psycopg2
from datetime import datetime, timedelta
from queries import QUERIES

class WorkloadGenerator:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()
        
        self.customer_range = self.get_range('customer', 'c_custkey')
        self.order_range = self.get_range('orders', 'o_orderkey')
        
    def get_range(self, table, column):
        self.cursor.execute(f"SELECT MIN({column}), MAX({column}) FROM {table}")
        return self.cursor.fetchone()
    
    def generate_params(self, query_name):
        
        if query_name in ['q1_orders_by_customer', 'q2_customer_info', 'q5_join_customer_orders']:
            return (random.randint(*self.customer_range),)
        
        elif query_name == 'q3_order_details':
            return (random.randint(*self.order_range),)
        
        elif query_name == 'q4_orders_in_range':
            start = random.randint(self.customer_range[0], self.customer_range[1] - 1000)
            return (start, start + 1000)
        
        elif query_name == 'q6_expensive_orders':
            return (random.uniform(100000, 500000),)
        
        elif query_name == 'q7_recent_orders':
            days_ago = random.randint(0, 365)
            date = datetime(1998, 8, 1) - timedelta(days=days_ago)
            return (date.strftime('%Y-%m-%d'),)
        
        return ()
    
    def execute_query(self, query_name, params=None):
        
        if params is None:
            params = self.generate_params(query_name)
        
        query = QUERIES[query_name]
        
        start_time = time.time()
        try:
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            execution_time = time.time() - start_time
            
            return {
                'query': query_name,
                'params': params,
                'execution_time': execution_time,
                'rows_returned': len(results),
                'success': True
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'query': query_name,
                'params': params,
                'execution_time': execution_time,
                'error': str(e),
                'success': False
            }
    
    def run_workload(self, num_queries=100, distribution='zipfian'):

        
        query_names = list(QUERIES.keys())
        results = []
        
        print(f"\nRunning {num_queries} queries with {distribution} distribution")
        
        for i in range(num_queries):
            if distribution == 'zipfian':
                if random.random() < 0.8:
                    query = random.choice(query_names[:2]) 
                else:
                    query = random.choice(query_names)
            else:
                query = random.choice(query_names)
            
            result = self.execute_query(query)
            results.append(result)
            
            
            time.sleep(0.01)
        
        print(f" Completed {num_queries} queries")
        return results
    
    def print_summary(self, results):        
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        avg_time = sum(r['execution_time'] for r in results) / total
        
        print(f"\nWorkload Summary:")
        print(f"Total Queries: {total}, Successful: {successful}, Failed: {total - successful}, Avg Execution Time: {avg_time:.4f}s")
        
        query_counts = {}
        for r in results:
            query_counts[r['query']] = query_counts.get(r['query'], 0) + 1
        
        print(f"\nQuery Distribution:")
        for query, count in sorted(query_counts.items(), key=lambda x: -x[1]):
            print(f"{query}: {count} ({count/total*100:.1f}%)")
    
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
    
    generator = WorkloadGenerator(DB_CONFIG)
    results = generator.run_workload(num_queries=50, distribution='zipfian')
    generator.print_summary(results)
    
    generator.close()