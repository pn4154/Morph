QUERIES = {
    'q1_orders_by_customer': """
        SELECT o_custkey, COUNT(*) as order_count
        FROM orders
        WHERE o_custkey = %s
        GROUP BY o_custkey;
    """,
    
    'q2_customer_info': """
        SELECT c_name, c_acctbal
        FROM customer
        WHERE c_custkey = %s;
    """,
    
    'q3_order_details': """
        SELECT o_orderkey, o_orderdate, o_totalprice
        FROM orders
        WHERE o_orderkey = %s;
    """,
    
    'q4_orders_in_range': """
        SELECT COUNT(*)
        FROM orders
        WHERE o_custkey BETWEEN %s AND %s;
    """,
    
    'q5_join_customer_orders': """
        SELECT c.c_name, COUNT(o.o_orderkey) as num_orders
        FROM customer c
        JOIN orders o ON c.c_custkey = o.o_custkey
        WHERE c.c_custkey = %s
        GROUP BY c.c_name;
    """,
    
    'q6_expensive_orders': """
        SELECT o_orderkey, o_totalprice
        FROM orders
        WHERE o_totalprice > %s
        LIMIT 10;
    """,
    
    'q7_recent_orders': """
        SELECT o_orderkey, o_orderdate
        FROM orders
        WHERE o_orderdate >= %s
        ORDER BY o_orderdate DESC
        LIMIT 20;
    """,
}