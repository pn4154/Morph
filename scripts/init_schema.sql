-- Morph TPC-H Schema Initialization
-- This script creates the TPC-H schema optimized for Citus distribution

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS citus;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- ============================================
-- Reference Tables (replicated across all nodes)
-- ============================================

-- Region table
CREATE TABLE IF NOT EXISTS region (
    r_regionkey INTEGER PRIMARY KEY,
    r_name CHAR(25) NOT NULL,
    r_comment VARCHAR(152)
);

-- Nation table
CREATE TABLE IF NOT EXISTS nation (
    n_nationkey INTEGER PRIMARY KEY,
    n_name CHAR(25) NOT NULL,
    n_regionkey INTEGER NOT NULL REFERENCES region(r_regionkey),
    n_comment VARCHAR(152)
);

-- ============================================
-- Distributed Tables
-- ============================================

-- Customer table (distributed by c_custkey)
CREATE TABLE IF NOT EXISTS customer (
    c_custkey INTEGER NOT NULL,
    c_name VARCHAR(25) NOT NULL,
    c_address VARCHAR(40) NOT NULL,
    c_nationkey INTEGER NOT NULL,
    c_phone CHAR(15) NOT NULL,
    c_acctbal DECIMAL(15,2) NOT NULL,
    c_mktsegment CHAR(10) NOT NULL,
    c_comment VARCHAR(117),
    PRIMARY KEY (c_custkey)
);

-- Supplier table (distributed by s_suppkey)
CREATE TABLE IF NOT EXISTS supplier (
    s_suppkey INTEGER NOT NULL,
    s_name CHAR(25) NOT NULL,
    s_address VARCHAR(40) NOT NULL,
    s_nationkey INTEGER NOT NULL,
    s_phone CHAR(15) NOT NULL,
    s_acctbal DECIMAL(15,2) NOT NULL,
    s_comment VARCHAR(101),
    PRIMARY KEY (s_suppkey)
);

-- Part table (distributed by p_partkey)
CREATE TABLE IF NOT EXISTS part (
    p_partkey INTEGER NOT NULL,
    p_name VARCHAR(55) NOT NULL,
    p_mfgr CHAR(25) NOT NULL,
    p_brand CHAR(10) NOT NULL,
    p_type VARCHAR(25) NOT NULL,
    p_size INTEGER NOT NULL,
    p_container CHAR(10) NOT NULL,
    p_retailprice DECIMAL(15,2) NOT NULL,
    p_comment VARCHAR(23),
    PRIMARY KEY (p_partkey)
);

-- Partsupp table (distributed by ps_partkey)
CREATE TABLE IF NOT EXISTS partsupp (
    ps_partkey INTEGER NOT NULL,
    ps_suppkey INTEGER NOT NULL,
    ps_availqty INTEGER NOT NULL,
    ps_supplycost DECIMAL(15,2) NOT NULL,
    ps_comment VARCHAR(199),
    PRIMARY KEY (ps_partkey, ps_suppkey)
);

-- Orders table (distributed by o_orderkey)
CREATE TABLE IF NOT EXISTS orders (
    o_orderkey BIGINT NOT NULL,
    o_custkey INTEGER NOT NULL,
    o_orderstatus CHAR(1) NOT NULL,
    o_totalprice DECIMAL(15,2) NOT NULL,
    o_orderdate DATE NOT NULL,
    o_orderpriority CHAR(15) NOT NULL,
    o_clerk CHAR(15) NOT NULL,
    o_shippriority INTEGER NOT NULL,
    o_comment VARCHAR(79),
    PRIMARY KEY (o_orderkey)
);

-- Lineitem table (distributed by l_orderkey, co-located with orders)
CREATE TABLE IF NOT EXISTS lineitem (
    l_orderkey BIGINT NOT NULL,
    l_partkey INTEGER NOT NULL,
    l_suppkey INTEGER NOT NULL,
    l_linenumber INTEGER NOT NULL,
    l_quantity DECIMAL(15,2) NOT NULL,
    l_extendedprice DECIMAL(15,2) NOT NULL,
    l_discount DECIMAL(15,2) NOT NULL,
    l_tax DECIMAL(15,2) NOT NULL,
    l_returnflag CHAR(1) NOT NULL,
    l_linestatus CHAR(1) NOT NULL,
    l_shipdate DATE NOT NULL,
    l_commitdate DATE NOT NULL,
    l_receiptdate DATE NOT NULL,
    l_shipinstruct CHAR(25) NOT NULL,
    l_shipmode CHAR(10) NOT NULL,
    l_comment VARCHAR(44),
    PRIMARY KEY (l_orderkey, l_linenumber)
);

-- ============================================
-- Create Reference Tables in Citus
-- ============================================

SELECT create_reference_table('region');
SELECT create_reference_table('nation');

-- ============================================
-- Distribute Tables with Optimal Shard Count
-- ============================================

-- Distribute customer table
SELECT create_distributed_table('customer', 'c_custkey', shard_count := 18);

-- Distribute supplier table
SELECT create_distributed_table('supplier', 's_suppkey', shard_count := 6);

-- Distribute part table
SELECT create_distributed_table('part', 'p_partkey', shard_count := 12);

-- Distribute partsupp table (co-locate with part)
SELECT create_distributed_table('partsupp', 'ps_partkey', colocate_with := 'part');

-- Distribute orders table
SELECT create_distributed_table('orders', 'o_orderkey', shard_count := 18);

-- Distribute lineitem table (co-locate with orders for efficient joins)
SELECT create_distributed_table('lineitem', 'l_orderkey', colocate_with := 'orders');

-- ============================================
-- Create Indexes for Query Performance
-- ============================================

-- Customer indexes
CREATE INDEX IF NOT EXISTS idx_customer_nationkey ON customer(c_nationkey);
CREATE INDEX IF NOT EXISTS idx_customer_mktsegment ON customer(c_mktsegment);

-- Supplier indexes
CREATE INDEX IF NOT EXISTS idx_supplier_nationkey ON supplier(s_nationkey);

-- Orders indexes
CREATE INDEX IF NOT EXISTS idx_orders_custkey ON orders(o_custkey);
CREATE INDEX IF NOT EXISTS idx_orders_orderdate ON orders(o_orderdate);

-- Lineitem indexes
CREATE INDEX IF NOT EXISTS idx_lineitem_partkey ON lineitem(l_partkey);
CREATE INDEX IF NOT EXISTS idx_lineitem_suppkey ON lineitem(l_suppkey);
CREATE INDEX IF NOT EXISTS idx_lineitem_shipdate ON lineitem(l_shipdate);
CREATE INDEX IF NOT EXISTS idx_lineitem_commitdate ON lineitem(l_commitdate);
CREATE INDEX IF NOT EXISTS idx_lineitem_receiptdate ON lineitem(l_receiptdate);

-- Part indexes
CREATE INDEX IF NOT EXISTS idx_part_type ON part(p_type);
CREATE INDEX IF NOT EXISTS idx_part_size ON part(p_size);

-- ============================================
-- Verification Queries
-- ============================================

-- Check distributed tables
SELECT logicalrelid, partmethod, partkey, colocationid, repmodel
FROM pg_dist_partition
ORDER BY logicalrelid;

-- Check shard distribution
SELECT 
    logicalrelid::text AS table_name,
    COUNT(*) AS shard_count
FROM pg_dist_shard
GROUP BY logicalrelid
ORDER BY logicalrelid;

-- Check shard placement per node
SELECT 
    nodename,
    COUNT(*) AS shard_count
FROM pg_dist_shard_placement
GROUP BY nodename
ORDER BY nodename;

-- Summary
DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'TPC-H Schema initialization complete!';
    RAISE NOTICE '============================================';
END $$;
