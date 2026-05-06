import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, round

# 1. Khoi tao Spark Session cho he thong xu ly luong
spark = SparkSession.builder \
    .appName("Realtime_Revenue_Monitoring_System") \
    .getOrCreate()

# 2. Thiet lap nguon du lieu gia lap (Data Source)
raw_stream_df = spark.readStream \
    .format("rate") \
    .option("rowsPerSecond", 1) \
    .load()

# 3. Tien xu ly va chuan hoa du lieu (Data Transformation)
processed_sales_df = raw_stream_df.select(
    current_timestamp().alias("timestamp"), 
    col("value").cast("string").alias("transaction_id"),
    round((col("value") % 100) * 1.5, 2).alias("revenue") 
)

# 4. CAU HINH DUONG DAN DAU RA (DA SUA CHO DOCKER)
# /home/jovyan/work tuong ung voi thu muc BCCK trong container, du lieu se duoc ghi vao thu muc data/realtime_output tren host
output_path = "/home/jovyan/work/data/realtime_output"
checkpoint_path = "/home/jovyan/work/checkpoints/streaming_v1"

# 5. Kich hoat luong ghi du lieu (Streaming Query)
streaming_query = processed_sales_df.writeStream \
    .format("csv") \
    .option("path", output_path) \
    .option("checkpointLocation", checkpoint_path) \
    .option("header", "true") \
    .outputMode("append") \
    .start()

print(f"--- [He thong dang chay] Du lieu dang duoc day ra folder: data/realtime_output ---")

# Duy tri luong xu ly
streaming_query.awaitTermination()