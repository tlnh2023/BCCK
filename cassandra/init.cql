-- ════════════════════════════════════════════════════════════════
--  Cassandra Init Script
--  File này tự động chạy khi container khởi động lần đầu
--  Vị trí: ./cassandra/init.cql
-- ════════════════════════════════════════════════════════════════

CREATE KEYSPACE IF NOT EXISTS ecommerce
  WITH replication = {
    'class': 'SimpleStrategy',
    'replication_factor': 1
  };

USE ecommerce;

-- Doanh thu theo brand (streaming ghi vào)
CREATE TABLE IF NOT EXISTS brand_revenue (
  brand          TEXT,
  month          TEXT,
  total_revenue  DOUBLE,
  order_count    BIGINT,
  updated_at     TIMESTAMP,
  PRIMARY KEY (brand, month)
);

-- Xu hướng danh mục sản phẩm
CREATE TABLE IF NOT EXISTS category_trend (
  category       TEXT,
  month          TEXT,
  event_count    BIGINT,
  PRIMARY KEY (category, month)
);

-- Kết quả phân cụm RFM (batch ghi vào)
CREATE TABLE IF NOT EXISTS customer_segments (
  user_id        TEXT PRIMARY KEY,
  recency        INT,
  frequency      INT,
  monetary       DOUBLE,
  cluster_id     INT,
  segment_name   TEXT,
  created_at     TIMESTAMP
);

-- Doanh thu theo tháng (cho line chart)
CREATE TABLE IF NOT EXISTS monthly_revenue (
  year           INT,
  month          INT,
  total_revenue  DOUBLE,
  unique_buyers  BIGINT,
  PRIMARY KEY (year, month)
);
