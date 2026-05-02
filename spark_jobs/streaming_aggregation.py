from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, DoubleType, IntegerType

# 1. Khởi tạo Spark Session - Kết nối thẳng tới container 'cassandra'
spark = SparkSession.builder \
    .appName("CustomerBehaviorStreaming") \
    .config("spark.cassandra.connection.host", "cassandra") \
    .config("spark.sql.extensions", "com.datastax.spark.connector.CassandraSparkExtensions") \
    .getOrCreate()

# 2. Đọc luồng dữ liệu giả lập (1 dòng/giây) để tạo data test
df = spark.readStream \
    .format("rate") \
    .option("rowsPerSecond", 1) \
    .load()

# 3. Tiền xử lý dữ liệu thô
processed_df = df.select(
    col("timestamp").cast("string"),
    col("value").cast("string").alias("transaction_id"), 
    col("value").cast("string").alias("brand"),          
    col("value").cast("double").alias("amount"),
    col("value").cast("int").alias("customer_id"),
    col("value").cast("string").alias("category")
)

# 4. Ghi vào bảng 'transactions' (Lưu trữ dữ liệu thô - Mục 4.3)
query_raw = processed_df.writeStream \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="transactions", keyspace="project_ks") \
    .option("checkpointLocation", "/home/jovyan/work/checkpoints/raw_data") \
    .outputMode("append") \
    .start()

# 5. CHỈ chọn các cột cần thiết cho bảng 'brand_stats' (Dùng cho Dashboard)
brand_stats_df = processed_df.select(
    col("brand"),
    col("amount").alias("total_revenue"),
    col("customer_id").alias("transaction_count") 
)

# 6. Ghi vào bảng 'brand_stats'
query_agg = brand_stats_df.writeStream \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="brand_stats", keyspace="project_ks") \
    .option("checkpointLocation", "/home/jovyan/work/checkpoints/agg_data") \
    .outputMode("append") \
    .start()

print(">>> Hệ thống đang đẩy dữ liệu vào Cassandra. Mở cqlsh kiểm tra ngay!")
spark.streams.awaitAnyTermination()