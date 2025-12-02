#!/usr/bin/env python3
"""
TPC-H Data Generator for Morph Project

Generates TPC-H benchmark data at configurable scale factors.
This is a pure Python implementation that generates data matching
the TPC-H specification.
"""

import argparse
import csv
import os
import random
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import string
import sys
from typing import List, Tuple, Generator
import psycopg2
from psycopg2.extras import execute_values

# TPC-H text pools
TEXT_POOL = [
    "the", "of", "and", "a", "to", "in", "is", "you", "that", "it",
    "he", "was", "for", "on", "are", "as", "with", "his", "they", "I",
    "at", "be", "this", "have", "from", "or", "one", "had", "by", "word",
    "but", "not", "what", "all", "were", "we", "when", "your", "can", "said",
    "there", "use", "an", "each", "which", "she", "do", "how", "their", "if",
    "will", "up", "other", "about", "out", "many", "then", "them", "these", "so",
    "some", "her", "would", "make", "like", "him", "into", "time", "has", "look",
    "two", "more", "write", "go", "see", "number", "no", "way", "could", "people",
    "my", "than", "first", "water", "been", "call", "who", "oil", "its", "now",
    "find", "long", "down", "day", "did", "get", "come", "made", "may", "part"
]

REGIONS = [
    (0, "AFRICA", "lar deposits. blithely final packages cajole. regular waters are final requests."),
    (1, "AMERICA", "hs use ironic, even requests. s"),
    (2, "ASIA", "ges. thinly even pinto beans ca"),
    (3, "EUROPE", "ly final courts cajole furiously final excuse"),
    (4, "MIDDLE EAST", "uickly special accounts cajole carefully blithely close requests.")
]

NATIONS = [
    (0, "ALGERIA", 0), (1, "ARGENTINA", 1), (2, "BRAZIL", 1), (3, "CANADA", 1),
    (4, "EGYPT", 4), (5, "ETHIOPIA", 0), (6, "FRANCE", 3), (7, "GERMANY", 3),
    (8, "INDIA", 2), (9, "INDONESIA", 2), (10, "IRAN", 4), (11, "IRAQ", 4),
    (12, "JAPAN", 2), (13, "JORDAN", 4), (14, "KENYA", 0), (15, "MOROCCO", 0),
    (16, "MOZAMBIQUE", 0), (17, "PERU", 1), (18, "CHINA", 2), (19, "ROMANIA", 3),
    (20, "SAUDI ARABIA", 4), (21, "VIETNAM", 2), (22, "RUSSIA", 3),
    (23, "UNITED KINGDOM", 3), (24, "UNITED STATES", 1)
]

SEGMENTS = ["AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"]

PRIORITIES = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]

SHIP_MODES = ["REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"]

SHIP_INSTRUCT = ["DELIVER IN PERSON", "COLLECT COD", "NONE", "TAKE BACK RETURN"]

TYPES = [
    "STANDARD ANODIZED TIN", "STANDARD ANODIZED NICKEL", "STANDARD ANODIZED BRASS",
    "STANDARD ANODIZED STEEL", "STANDARD ANODIZED COPPER", "STANDARD BURNISHED TIN",
    "STANDARD BURNISHED NICKEL", "STANDARD BURNISHED BRASS", "STANDARD BURNISHED STEEL",
    "STANDARD BURNISHED COPPER", "STANDARD PLATED TIN", "STANDARD PLATED NICKEL",
    "STANDARD PLATED BRASS", "STANDARD PLATED STEEL", "STANDARD PLATED COPPER",
    "STANDARD POLISHED TIN", "STANDARD POLISHED NICKEL", "STANDARD POLISHED BRASS",
    "STANDARD POLISHED STEEL", "STANDARD POLISHED COPPER", "STANDARD BRUSHED TIN",
    "STANDARD BRUSHED NICKEL", "STANDARD BRUSHED BRASS", "STANDARD BRUSHED STEEL",
    "STANDARD BRUSHED COPPER", "SMALL ANODIZED TIN", "SMALL ANODIZED NICKEL",
    "SMALL ANODIZED BRASS", "SMALL ANODIZED STEEL", "SMALL ANODIZED COPPER"
]

CONTAINERS = [
    "SM CASE", "SM BOX", "SM BAG", "SM JAR", "SM PACK", "SM PKG", "SM CAN", "SM DRUM",
    "MED CASE", "MED BOX", "MED BAG", "MED JAR", "MED PACK", "MED PKG", "MED CAN", "MED DRUM",
    "LG CASE", "LG BOX", "LG BAG", "LG JAR", "LG PACK", "LG PKG", "LG CAN", "LG DRUM",
    "JUMBO CASE", "JUMBO BOX", "JUMBO BAG", "JUMBO JAR", "JUMBO PACK", "JUMBO PKG", "JUMBO CAN", "JUMBO DRUM",
    "WRAP CASE", "WRAP BOX", "WRAP BAG", "WRAP JAR", "WRAP PACK", "WRAP PKG", "WRAP CAN", "WRAP DRUM"
]

BRANDS = [f"Brand#{i}{j}" for i in range(1, 6) for j in range(1, 6)]


def random_text(min_len: int, max_len: int) -> str:
    """Generate random TPC-H style text"""
    length = random.randint(min_len, max_len)
    words = []
    current_len = 0
    while current_len < length:
        word = random.choice(TEXT_POOL)
        words.append(word)
        current_len += len(word) + 1
    return ' '.join(words)[:length]


def random_phone(nation_key: int) -> str:
    """Generate TPC-H phone number"""
    country_code = 10 + nation_key
    return f"{country_code}-{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"


class TPCHGenerator:
    """Generator for TPC-H benchmark data"""
    
    def __init__(self, scale_factor: float = 1.0, seed: int = 42):
        self.scale_factor = scale_factor
        self.seed = seed
        random.seed(seed)
        
        # Table sizes based on scale factor
        self.n_suppliers = int(10000 * scale_factor)
        self.n_parts = int(200000 * scale_factor)
        self.n_customers = int(150000 * scale_factor)
        self.n_orders = int(1500000 * scale_factor)
        
        # Date range for orders
        self.start_date = date(1992, 1, 1)
        self.end_date = date(1998, 12, 31)
        self.current_date = date(1995, 6, 17)
    
    def generate_regions(self) -> List[Tuple]:
        """Generate region table data"""
        return [
            (r[0], r[1].ljust(25), r[2])
            for r in REGIONS
        ]
    
    def generate_nations(self) -> List[Tuple]:
        """Generate nation table data"""
        return [
            (n[0], n[1].ljust(25), n[2], random_text(31, 114))
            for n in NATIONS
        ]
    
    def generate_suppliers(self) -> Generator[Tuple, None, None]:
        """Generate supplier table data"""
        for suppkey in range(1, self.n_suppliers + 1):
            nationkey = random.randint(0, 24)
            yield (
                suppkey,
                f"Supplier#{suppkey:09d}".ljust(25),
                random_text(10, 40),
                nationkey,
                random_phone(nationkey),
                Decimal(random.randint(-99999, 999999)) / 100,
                random_text(25, 100)
            )
    
    def generate_parts(self) -> Generator[Tuple, None, None]:
        """Generate part table data"""
        for partkey in range(1, self.n_parts + 1):
            yield (
                partkey,
                random_text(22, 55),
                f"Manufacturer#{random.randint(1,5)}".ljust(25),
                random.choice(BRANDS).ljust(10),
                random.choice(TYPES),
                random.randint(1, 50),
                random.choice(CONTAINERS).ljust(10),
                Decimal(random.randint(100, 200000)) / 100,
                random_text(5, 22)
            )
    
    def generate_partsupp(self) -> Generator[Tuple, None, None]:
        """Generate partsupp table data"""
        for partkey in range(1, self.n_parts + 1):
            # Each part has 4 suppliers
            suppliers = random.sample(range(1, self.n_suppliers + 1), min(4, self.n_suppliers))
            for suppkey in suppliers:
                yield (
                    partkey,
                    suppkey,
                    random.randint(1, 9999),
                    Decimal(random.randint(100, 100000)) / 100,
                    random_text(49, 198)
                )
    
    def generate_customers(self) -> Generator[Tuple, None, None]:
        """Generate customer table data"""
        for custkey in range(1, self.n_customers + 1):
            nationkey = random.randint(0, 24)
            yield (
                custkey,
                f"Customer#{custkey:09d}",
                random_text(10, 40),
                nationkey,
                random_phone(nationkey),
                Decimal(random.randint(-99999, 999999)) / 100,
                random.choice(SEGMENTS).ljust(10),
                random_text(29, 116)
            )
    
    def generate_orders_and_lineitems(self) -> Tuple[Generator[Tuple, None, None], Generator[Tuple, None, None]]:
        """Generate orders and lineitem table data"""
        orders = []
        lineitems = []
        
        date_range = (self.end_date - self.start_date).days
        
        for orderkey in range(1, self.n_orders + 1):
            custkey = random.randint(1, self.n_customers)
            orderdate = self.start_date + timedelta(days=random.randint(0, date_range))
            
            # Generate 1-7 lineitems per order
            n_lineitems = random.randint(1, 7)
            total_price = Decimal(0)
            
            order_lineitems = []
            for linenumber in range(1, n_lineitems + 1):
                partkey = random.randint(1, self.n_parts)
                suppkey = random.randint(1, self.n_suppliers)
                quantity = Decimal(random.randint(1, 50))
                extendedprice = Decimal(random.randint(90000, 110000)) / 100 * quantity
                discount = Decimal(random.randint(0, 10)) / 100
                tax = Decimal(random.randint(0, 8)) / 100
                
                total_price += extendedprice * (1 - discount) * (1 + tax)
                
                shipdate = orderdate + timedelta(days=random.randint(1, 121))
                commitdate = orderdate + timedelta(days=random.randint(30, 90))
                receiptdate = shipdate + timedelta(days=random.randint(1, 30))
                
                # Determine return flag and line status
                if receiptdate <= self.current_date:
                    returnflag = random.choice(['R', 'A', 'N'])
                    linestatus = 'F'
                else:
                    returnflag = 'N'
                    linestatus = 'O'
                
                order_lineitems.append((
                    orderkey,
                    partkey,
                    suppkey,
                    linenumber,
                    quantity,
                    extendedprice,
                    discount,
                    tax,
                    returnflag,
                    linestatus,
                    shipdate,
                    commitdate,
                    receiptdate,
                    random.choice(SHIP_INSTRUCT).ljust(25),
                    random.choice(SHIP_MODES).ljust(10),
                    random_text(10, 43)
                ))
            
            # Determine order status
            linestatus_set = set(li[9] for li in order_lineitems)
            if linestatus_set == {'F'}:
                orderstatus = 'F'
            elif linestatus_set == {'O'}:
                orderstatus = 'O'
            else:
                orderstatus = 'P'
            
            orders.append((
                orderkey,
                custkey,
                orderstatus,
                total_price.quantize(Decimal('0.01')),
                orderdate,
                random.choice(PRIORITIES).ljust(15),
                f"Clerk#{random.randint(1, 1000):09d}".ljust(15),
                0,  # shippriority
                random_text(19, 78)
            ))
            
            lineitems.extend(order_lineitems)
        
        return orders, lineitems


def load_to_database(
    generator: TPCHGenerator,
    host: str = 'localhost',
    port: int = 5432,
    database: str = 'morphdb',
    user: str = 'morph',
    password: str = 'MorphSecurePass123!',
    batch_size: int = 10000
):
    """Load generated data directly to database"""
    
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    print("Loading TPC-H data to database...")
    
    # Load regions
    print("Loading regions...")
    regions = generator.generate_regions()
    execute_values(cursor, 
        "INSERT INTO region (r_regionkey, r_name, r_comment) VALUES %s ON CONFLICT DO NOTHING",
        regions
    )
    
    # Load nations
    print("Loading nations...")
    nations = generator.generate_nations()
    execute_values(cursor,
        "INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES %s ON CONFLICT DO NOTHING",
        nations
    )
    
    # Load suppliers
    print(f"Loading {generator.n_suppliers} suppliers...")
    batch = []
    for supplier in generator.generate_suppliers():
        batch.append(supplier)
        if len(batch) >= batch_size:
            execute_values(cursor,
                "INSERT INTO supplier (s_suppkey, s_name, s_address, s_nationkey, s_phone, s_acctbal, s_comment) VALUES %s",
                batch
            )
            batch = []
    if batch:
        execute_values(cursor,
            "INSERT INTO supplier (s_suppkey, s_name, s_address, s_nationkey, s_phone, s_acctbal, s_comment) VALUES %s",
            batch
        )
    
    # Load parts
    print(f"Loading {generator.n_parts} parts...")
    batch = []
    for part in generator.generate_parts():
        batch.append(part)
        if len(batch) >= batch_size:
            execute_values(cursor,
                "INSERT INTO part (p_partkey, p_name, p_mfgr, p_brand, p_type, p_size, p_container, p_retailprice, p_comment) VALUES %s",
                batch
            )
            batch = []
    if batch:
        execute_values(cursor,
            "INSERT INTO part (p_partkey, p_name, p_mfgr, p_brand, p_type, p_size, p_container, p_retailprice, p_comment) VALUES %s",
            batch
        )
    
    # Load partsupp
    print("Loading partsupp...")
    batch = []
    for ps in generator.generate_partsupp():
        batch.append(ps)
        if len(batch) >= batch_size:
            execute_values(cursor,
                "INSERT INTO partsupp (ps_partkey, ps_suppkey, ps_availqty, ps_supplycost, ps_comment) VALUES %s",
                batch
            )
            batch = []
    if batch:
        execute_values(cursor,
            "INSERT INTO partsupp (ps_partkey, ps_suppkey, ps_availqty, ps_supplycost, ps_comment) VALUES %s",
            batch
        )
    
    # Load customers
    print(f"Loading {generator.n_customers} customers...")
    batch = []
    for customer in generator.generate_customers():
        batch.append(customer)
        if len(batch) >= batch_size:
            execute_values(cursor,
                "INSERT INTO customer (c_custkey, c_name, c_address, c_nationkey, c_phone, c_acctbal, c_mktsegment, c_comment) VALUES %s",
                batch
            )
            batch = []
    if batch:
        execute_values(cursor,
            "INSERT INTO customer (c_custkey, c_name, c_address, c_nationkey, c_phone, c_acctbal, c_mktsegment, c_comment) VALUES %s",
            batch
        )
    
    # Load orders and lineitems
    print(f"Loading {generator.n_orders} orders and lineitems...")
    orders, lineitems = generator.generate_orders_and_lineitems()
    
    # Load orders
    batch = []
    for i, order in enumerate(orders):
        batch.append(order)
        if len(batch) >= batch_size:
            execute_values(cursor,
                "INSERT INTO orders (o_orderkey, o_custkey, o_orderstatus, o_totalprice, o_orderdate, o_orderpriority, o_clerk, o_shippriority, o_comment) VALUES %s",
                batch
            )
            batch = []
            print(f"  Orders: {i+1}/{len(orders)}")
    if batch:
        execute_values(cursor,
            "INSERT INTO orders (o_orderkey, o_custkey, o_orderstatus, o_totalprice, o_orderdate, o_orderpriority, o_clerk, o_shippriority, o_comment) VALUES %s",
            batch
        )
    
    # Load lineitems
    print(f"Loading {len(lineitems)} lineitems...")
    batch = []
    for i, lineitem in enumerate(lineitems):
        batch.append(lineitem)
        if len(batch) >= batch_size:
            execute_values(cursor,
                """INSERT INTO lineitem (l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity, 
                   l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate,
                   l_commitdate, l_receiptdate, l_shipinstruct, l_shipmode, l_comment) VALUES %s""",
                batch
            )
            batch = []
            if i % 100000 == 0:
                print(f"  Lineitems: {i}/{len(lineitems)}")
    if batch:
        execute_values(cursor,
            """INSERT INTO lineitem (l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity, 
               l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate,
               l_commitdate, l_receiptdate, l_shipinstruct, l_shipmode, l_comment) VALUES %s""",
            batch
        )
    
    # Verify load
    print("\nVerifying data load...")
    tables = ['region', 'nation', 'supplier', 'part', 'partsupp', 'customer', 'orders', 'lineitem']
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count:,} rows")
    
    cursor.close()
    conn.close()
    print("\nData load complete!")


def main():
    parser = argparse.ArgumentParser(description='Generate TPC-H data for Morph')
    parser.add_argument('--scale', type=float, default=0.1,
                       help='Scale factor (0.1 = ~100MB, 1.0 = ~1GB)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Database host')
    parser.add_argument('--port', type=int, default=5432,
                       help='Database port')
    parser.add_argument('--database', type=str, default='morphdb',
                       help='Database name')
    parser.add_argument('--user', type=str, default='morph',
                       help='Database user')
    parser.add_argument('--password', type=str, default='MorphSecurePass123!',
                       help='Database password')
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='Batch size for inserts')
    
    args = parser.parse_args()
    
    print(f"TPC-H Data Generator")
    print(f"====================")
    print(f"Scale Factor: {args.scale}")
    print(f"Seed: {args.seed}")
    print()
    
    generator = TPCHGenerator(scale_factor=args.scale, seed=args.seed)
    
    print(f"Estimated row counts:")
    print(f"  Suppliers: {generator.n_suppliers:,}")
    print(f"  Parts: {generator.n_parts:,}")
    print(f"  Customers: {generator.n_customers:,}")
    print(f"  Orders: {generator.n_orders:,}")
    print()
    
    load_to_database(
        generator,
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
