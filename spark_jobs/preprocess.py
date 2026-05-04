"""
spark_jobs/preprocess.py
Chạy: spark-submit /home/jovyan/work/spark_jobs/preprocess.py
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, when, isnull, sum as _sum,
    round as _round, desc, countDistinct,
    avg, stddev, percentile_approx,
    to_timestamp, to_date, year, month,
    lower, trim, regexp_replace, split, lit
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType
)
import time, warnings, sys
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────
DATA_PATH = "/home/jovyan/work/data/kz.csv"
OUT_CLEAN = "/home/jovyan/work/data/clean_data/"
OUT_RFM   = "/home/jovyan/work/data/rfm_input/"

def main():
    # ── 1. Spark Session ─────────────────────────────────────────
    spark = SparkSession.builder \
        .appName("eCommerce_Preprocessing_Job") \
        .config("spark.driver.memory", "6g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.default.parallelism", "4") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark: {spark.version}  |  Python: {sys.version.split()[0]}")

    # ── 2. Đọc dữ liệu ───────────────────────────────────────────
    schema = StructType([
        StructField("event_time",    StringType(), True),
        StructField("order_id",      StringType(), True),
        StructField("product_id",    StringType(), True),
        StructField("category_id",   StringType(), True),
        StructField("category_code", StringType(), True),
        StructField("brand",         StringType(), True),
        StructField("price",         DoubleType(), True),
        StructField("user_id",       StringType(), True),
    ])

    t0 = time.time()
    df_raw = spark.read.csv(DATA_PATH, header=True, schema=schema)
    total_rows = df_raw.count()
    print(f"[1] Đọc xong: {total_rows:,} dòng  ({time.time()-t0:.1f}s)")

    # ── 3. Kiểm tra NULL (1 lần quét) ────────────────────────────
    null_counts = df_raw.select(
        _sum(when(isnull(col("brand")),         1).otherwise(0)).alias("brand_null"),
        _sum(when(isnull(col("category_code")), 1).otherwise(0)).alias("cat_null"),
        _sum(when(isnull(col("category_id")),   1).otherwise(0)).alias("catid_null"),
        _sum(when(isnull(col("price")),         1).otherwise(0)).alias("price_null"),
        _sum(when(isnull(col("user_id")),       1).otherwise(0)).alias("userid_null"),
        _sum(when(col("price") == 0,    1).otherwise(0)).alias("price_zero"),
        _sum(when(col("price") <  0,    1).otherwise(0)).alias("price_neg"),
        _sum(when(col("price") > 10000, 1).otherwise(0)).alias("price_high"),
    ).collect()[0]

    print(f"\n[2] Kiểm tra dữ liệu:")
    print(f"  brand NULL        : {null_counts['brand_null']:,}")
    print(f"  category_code NULL: {null_counts['cat_null']:,}")
    print(f"  category_id NULL  : {null_counts['catid_null']:,}")
    print(f"  price NULL        : {null_counts['price_null']:,}")
    print(f"  user_id NULL      : {null_counts['userid_null']:,}  (khách vãng lai)")
    print(f"  Giá = 0           : {null_counts['price_zero']:,}")
    print(f"  Giá âm            : {null_counts['price_neg']:,}")
    print(f"  Giá > 10000       : {null_counts['price_high']:,}")

    # ── 4. Làm sạch ──────────────────────────────────────────────
    t0 = time.time()
    df_clean = df_raw \
        .dropDuplicates() \
        .withColumn("event_time",    to_timestamp(col("event_time"))) \
        .withColumn("price",         col("price").cast("double")) \
        .withColumn("user_id",       col("user_id").cast("string")) \
        .withColumn("order_id",      col("order_id").cast("string")) \
        .withColumn("product_id",    col("product_id").cast("string")) \
        .withColumn("category_id",   col("category_id").cast("string")) \
        .withColumn("brand",         col("brand").cast("string")) \
        .withColumn("category_code", col("category_code").cast("string")) \
        .filter(col("event_time").isNotNull()) \
        .filter(col("price") > 0) \
        .withColumn("brand",
            when(isnull(col("brand")) | (trim(col("brand")) == ""),
                 lit("unknown")).otherwise(col("brand"))) \
        .withColumn("category_code",
            when(isnull(col("category_code")) | (trim(col("category_code")) == ""),
                 lit("unknown")).otherwise(col("category_code"))) \
        .withColumn("category_id",
            when(isnull(col("category_id")),
                 lit("0")).otherwise(col("category_id"))) \
        .withColumn("brand",
            lower(trim(regexp_replace(col("brand"), "[^a-zA-Z0-9 ]", "")))) \
        .withColumn("category_code",
            lower(trim(regexp_replace(col("category_code"), r"[^a-zA-Z0-9._]", "")))) \
        .withColumn("date",  to_date(col("event_time"))) \
        .withColumn("year",  year(col("event_time"))) \
        .withColumn("month", month(col("event_time"))) \
        .withColumn("category_lv1",
            when(col("category_code") != "unknown",
                 split(col("category_code"), r"\.").getItem(0))
            .otherwise(lit("unknown"))) \
        .withColumn("category_lv2",
            when(col("category_code") != "unknown",
                 split(col("category_code"), r"\.").getItem(1))
            .otherwise(lit("unknown"))) \
        .filter(col("year") > 2000)   # loại dòng bị lỗi năm 1970

    rows_clean = df_clean.count()
    print(f"\n[3] Làm sạch xong: {rows_clean:,} dòng  ({time.time()-t0:.1f}s)")
    print(f"    Loại bỏ: {total_rows - rows_clean:,} dòng")

    # ── 5. Tách luồng RFM ────────────────────────────────────────
    df_rfm = df_clean.filter(col("user_id").isNotNull())
    print(f"[4] df_rfm (có user_id): {df_rfm.count():,} dòng")

    # ── 6. Lưu Parquet ───────────────────────────────────────────
    t0 = time.time()
    df_clean.write.mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet(OUT_CLEAN)
    print(f"[5] clean_data: {time.time()-t0:.1f}s  →  {OUT_CLEAN}")

    t0 = time.time()
    df_rfm.write.mode("overwrite").parquet(OUT_RFM)
    print(f"[5] rfm_input : {time.time()-t0:.1f}s  →  {OUT_RFM}")

    print("\n[OK] Preprocessing hoàn tất!")
    spark.stop()

if __name__ == "__main__":
    main()