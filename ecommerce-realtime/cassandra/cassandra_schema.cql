-- cassandra/init.cql
CREATE KEYSPACE IF NOT EXISTS ecommerce
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE ecommerce;

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    order_id UUID PRIMARY KEY,
    user_id TEXT,
    product_id TEXT,
    product_name TEXT,
    category TEXT,
    quantity INT,
    price DECIMAL,
    total_amount DECIMAL,
    order_status TEXT,
    order_timestamp TIMESTAMP,
    payment_method TEXT
);

-- Real-time sales by category
CREATE TABLE IF NOT EXISTS sales_by_category (
    category TEXT,
    date DATE,
    total_sales DECIMAL,
    order_count INT,
    PRIMARY KEY (category, date)
) WITH CLUSTERING ORDER BY (date DESC);

-- Real-time top products
CREATE TABLE IF NOT EXISTS top_products (
    product_id TEXT,
    product_name TEXT,
    category TEXT,
    total_quantity INT,
    last_updated TIMESTAMP,
    PRIMARY KEY (product_id)
);

-- Hourly statistics
CREATE TABLE IF NOT EXISTS hourly_stats (
    hour TIMESTAMP,
    total_revenue DECIMAL,
    total_orders INT,
    avg_order_value DECIMAL,
    PRIMARY KEY (hour)
);

-- Materialized view for user orders
CREATE MATERIALIZED VIEW IF NOT EXISTS user_orders AS
SELECT user_id, order_id, order_timestamp, total_amount, order_status
FROM orders
WHERE user_id IS NOT NULL AND order_id IS NOT NULL
PRIMARY KEY (user_id, order_timestamp, order_id);

-- Insert sample data
INSERT INTO orders (order_id, user_id, product_id, product_name, category, quantity, price, total_amount, order_status, order_timestamp, payment_method)
VALUES (uuid(), 'user_001', 'prod_001', 'Gaming Laptop', 'Electronics', 1, 999.99, 999.99, 'COMPLETED', toTimestamp(now()), 'CREDIT_CARD');

INSERT INTO orders (order_id, user_id, product_id, product_name, category, quantity, price, total_amount, order_status, order_timestamp, payment_method)
VALUES (uuid(), 'user_002', 'prod_002', 'Wireless Mouse', 'Electronics', 2, 29.99, 59.98, 'COMPLETED', toTimestamp(now()), 'PAYPAL');

INSERT INTO orders (order_id, user_id, product_id, product_name, category, quantity, price, total_amount, order_status, order_timestamp, payment_method)
VALUES (uuid(), 'user_003', 'prod_003', 'Running Shoes', 'Sports', 1, 89.99, 89.99, 'PENDING', toTimestamp(now()), 'CASH');
