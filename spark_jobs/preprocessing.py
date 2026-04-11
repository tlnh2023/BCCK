"""
spark_jobs/preprocess.py
========================
Spark job tiền xử lý dữ liệu giao dịch (~2.6 triệu dòng) từ kz.csv.

Chạy:
    spark-submit spark_jobs/preprocess.py \
        --input  /data/kz.csv \
        --output /data/processed \
        [--partitions 200]

Các bước:
  1. Đọc CSV với schema tường minh
  2. Xử lý Missing Values (drop / fillna)
  3. Xóa Duplicate
  4. Chuyển đổi kiểu dữ liệu (timestamp, double, string)
  5. Tạo cột mới  (total_amount, date, year_month, category_level1)
  6. Chuẩn hóa chuỗi (brand, category → lowercase + trim)
  7. Spark ML Pipeline  (StringIndexer → VectorAssembler → StandardScaler)
  8. Lưu kết quả Parquet, phân vùng theo year_month
"""

import argparse
import logging
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, TimestampType
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, VectorAssembler, StandardScaler
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("preprocess")


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────
RAW_SCHEMA = StructType([
    StructField("event_time",    StringType(),  True),
    StructField("order_id",      StringType(),  True),
    StructField("product_id",    StringType(),  True),
    StructField("category_id",   StringType(),  True),
    StructField("category_code", StringType(),  True),
    StructField("brand",         StringType(),  True),
    StructField("price",         DoubleType(),  True),
    StructField("user_id",       StringType(),  True),
])

# Cột dùng làm khoá dedup
DEDUP_KEYS = ["event_time", "order_id", "product_id", "user_id"]

# Cột đưa vào ML Pipeline
FEATURE_COLS = ["price", "total_amount", "quantity",
                "brand_idx", "category_idx"]

# Cột ghi ra Parquet
OUTPUT_COLS = [
    "user_id", "order_id", "product_id",
    "event_time", "date", "year_month",
    "brand", "brand_idx",
    "category", "category_level1", "category_idx",
    "price", "quantity", "total_amount",
    "features_scaled",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def _count(df, label: str) -> int:
    n = df.count()
    log.info("%-35s %10s dòng", label + ":", f"{n:,}")
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def build_spark(partitions: int) -> SparkSession:
    return (SparkSession.builder
            .appName("BCCK_Preprocessing")
            .config("spark.sql.shuffle.partitions", str(partitions))
            .config("spark.driver.memory", "4g")
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
            .getOrCreate())


def read_raw(spark: SparkSession, path: str):
    log.info("Đọc CSV: %s", path)
    return (spark.read
            .option("header", True)
            .option("inferSchema", False)
            .schema(RAW_SCHEMA)
            .csv(path))


def handle_missing(df):
    """
    Xử lý Missing Values:
      - Xóa dòng thiếu price hoặc user_id (không có ý nghĩa phân tích)
      - Điền "unknown" / "0" cho brand, category_code, category_id
    """
    df = df.dropna(subset=["price", "user_id"])
    df = df.fillna({
        "brand":         "unknown",
        "category_code": "unknown",
        "category_id":   "0",
    })
    return df


def remove_duplicates(df):
    return df.dropDuplicates(DEDUP_KEYS)


def cast_types(df):
    """Chuyển đổi kiểu dữ liệu."""
    return (df
        # event_time: "2020-04-24 11:50:39 UTC" → TimestampType
        .withColumn(
            "event_time",
            F.to_timestamp(
                F.regexp_replace("event_time", r" UTC$", ""),
                "yyyy-MM-dd HH:mm:ss"
            )
        )
        .withColumn("price",      F.col("price").cast(DoubleType()))
        .withColumn("user_id",    F.col("user_id").cast(StringType()))
        .withColumn("order_id",   F.col("order_id").cast(StringType()))
        .withColumn("product_id", F.col("product_id").cast(StringType()))
    )


def engineer_features(df):
    """Tạo cột mới và chuẩn hóa chuỗi."""
    return (df
        # quantity: dataset không có sẵn → mặc định 1
        .withColumn("quantity", F.lit(1).cast("int"))
        # Cột tổng tiền
        .withColumn("total_amount", F.col("price") * F.col("quantity"))
        # Cột ngày (dùng cho RFM)
        .withColumn("date", F.date_trunc("day", F.col("event_time")).cast("date"))
        # Cột năm-tháng (dùng để partition)
        .withColumn("year_month", F.date_format("event_time", "yyyy-MM"))
        # Chuẩn hóa brand & category
        .withColumn("brand",    F.trim(F.lower(F.col("brand"))))
        .withColumn("category", F.trim(F.lower(F.col("category_code"))))
        # Category cấp 1: "electronics.tablet" → "electronics"
        .withColumn("category_level1",
                    F.split(F.col("category"), r"\.").getItem(0))
        # Lọc giá hợp lệ
        .filter(F.col("price") > 0)
    )


def build_ml_pipeline() -> Pipeline:
    """
    ML Pipeline:
      StringIndexer (brand, category) → VectorAssembler → StandardScaler
    """
    brand_idx = StringIndexer(
        inputCol="brand", outputCol="brand_idx", handleInvalid="keep"
    )
    cat_idx = StringIndexer(
        inputCol="category", outputCol="category_idx", handleInvalid="keep"
    )
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="features_raw",
        handleInvalid="keep",
    )
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features_scaled",
        withMean=True,
        withStd=True,
    )
    return Pipeline(stages=[brand_idx, cat_idx, assembler, scaler])


def save_parquet(df, output_path: str):
    log.info("Lưu Parquet → %s (partition: year_month)", output_path)
    (df
     .write
     .mode("overwrite")
     .partitionBy("year_month")
     .parquet(output_path))
    log.info(" Lưu thành công.")


def run(input_path: str, output_path: str, partitions: int = 200):
    t0 = time.time()
    spark = build_spark(partitions)
    spark.sparkContext.setLogLevel("WARN")

    # ── 1. Đọc dữ liệu ───────────────────────────────────────────────────────
    df = read_raw(spark, input_path)
    n_raw = _count(df, "Raw")

    # ── 2. Missing values ─────────────────────────────────────────────────────
    df = handle_missing(df)
    n_null = _count(df, "Sau xử lý null")

    # ── 3. Duplicate ──────────────────────────────────────────────────────────
    df = remove_duplicates(df)
    n_dedup = _count(df, "Sau xử lý duplicate")

    # ── 4 & 5 & 6. Kiểu dữ liệu + Feature engineering ────────────────────────
    df = cast_types(df)
    df = engineer_features(df)
    n_clean = _count(df, "Sau lọc giá hợp lệ")

    # ── 7. ML Pipeline ────────────────────────────────────────────────────────
    log.info("Fit ML Pipeline …")
    pipeline_model = build_ml_pipeline().fit(df)
    df = pipeline_model.transform(df)
    log.info("ML Pipeline hoàn thành.")

    # ── 8. Lưu Parquet ────────────────────────────────────────────────────────
    df_out = df.select(*OUTPUT_COLS)
    save_parquet(df_out, output_path)

    elapsed = time.time() - t0
    log.info("=" * 55)
    log.info("  TỔNG KẾT")
    log.info("  Raw           : %10s dòng", f"{n_raw:,}")
    log.info("  Sau null-drop : %10s dòng", f"{n_null:,}")
    log.info("  Sau dedup     : %10s dòng", f"{n_dedup:,}")
    log.info("  Output        : %10s dòng", f"{n_clean:,}")
    log.info("  Thời gian     : %.1f giây", elapsed)
    log.info("=" * 55)

    spark.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCCK Preprocessing Spark Job")
    parser.add_argument("--input",      required=True,
                        help="Đường dẫn file CSV đầu vào (kz.csv)")
    parser.add_argument("--output",     required=True,
                        help="Thư mục Parquet đầu ra")
    parser.add_argument("--partitions", type=int, default=200,
                        help="spark.sql.shuffle.partitions (mặc định 200)")
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        partitions=args.partitions,
    )
