import psycopg2
import json
import time
from datetime import datetime
from collections import defaultdict
import os

class AccessPatternMonitor:
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.access_log = []  
        self.access_frequency = defaultdict(int)
        self.co_access_pairs = defaultdict(int)
        
    def track_query(self, query_text, params, execution_time, rows_returned):
        accessed_ranges = self._extract_accessed_ranges(query_text, params)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query_text[:100],
            'params': str(params),
            'execution_time': execution_time,
            'rows_returned': rows_returned,
            'accessed_ranges': accessed_ranges
        }
        
        self.access_log.append(log_entry)
        
        for range_id in accessed_ranges:
            self.access_frequency[range_id] += 1
        
        if len(accessed_ranges) > 1:
            for i, range1 in enumerate(accessed_ranges):
                for range2 in accessed_ranges[i+1:]:
                    pair = tuple(sorted([range1, range2]))
                    self.co_access_pairs[pair] += 1
        
        return log_entry
    
    def _extract_accessed_ranges(self, query_text, params):
        accessed = []
        if params:
            for param in params:
                if isinstance(param, int):
                    range_id = f"range_{(param // 1000) * 1000}-{((param // 1000) + 1) * 1000}"
                    accessed.append(range_id)
        
        return accessed if accessed else ['unknown']
    
    def get_access_heatmap(self):
        return dict(self.access_frequency)
    
    def get_co_access_patterns(self, min_count=5):
        return {pair: count for pair, count in self.co_access_pairs.items() 
                if count >= min_count}
    
    def print_summary(self):        
        print("Access pattern summary:")        
        print(f"\nTotal queries tracked: {len(self.access_log)}")
        print(f"Data ranges accessed: {len(self.access_frequency)}")
        
        print("top data ranges")
        sorted_ranges = sorted(self.access_frequency.items(), 
                              key=lambda x: -x[1])[:10]
        for range_id, count in sorted_ranges:
            print(f"  {range_id}: {count} accesses")
        
        print("co-access patterns")
        sorted_pairs = sorted(self.co_access_pairs.items(), 
                             key=lambda x: -x[1])[:5]
        for pair, count in sorted_pairs:
            print(f"  {pair[0]} + {pair[1]}: {count} times")
        
        if self.access_log:
            avg_time = sum(log['execution_time'] for log in self.access_log) / len(self.access_log)
            print(f"\nAvg query time: {avg_time:.4f} secs")


    def save_to_file(self, filename='./logs/access_patterns.json'):
        """Save all tracked data to a file"""
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        data = {
            'access_log': self.access_log,
            'access_frequency': dict(self.access_frequency),
            'co_access_pairs': {str(k): v for k, v in self.co_access_pairs.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        

if __name__ == "__main__":
    monitor = AccessPatternMonitor(None)    
    monitor.track_query("SELECT * FROM orders WHERE o_custkey = %s", (5,), 0.023, 10)
    monitor.track_query("SELECT * FROM orders WHERE o_custkey = %s", (5,), 0.018, 10)
    monitor.track_query("SELECT * FROM orders WHERE o_custkey = %s", (1005,), 0.045, 8)
    monitor.track_query("SELECT * FROM orders WHERE o_custkey = %s", (5,), 0.021, 10)
    monitor.print_summary()