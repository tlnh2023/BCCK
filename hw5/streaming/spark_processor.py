"""Apache Spark streaming processor for social media data"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType
from dotenv import load_dotenv

load_dotenv()


class SparkProcessor:
    """Spark streaming processor for Kafka topics"""
    
    def __init__(self, app_name: str = "SocialMediaProcessor"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .getOrCreate()
        
        self.brokers = os.getenv("KAFKA_BROKERS", "localhost:9092")
    
    def create_schema(self):
        """Define schema for streaming data"""
        return StructType([
            StructField("source", StringType()),
            StructField("id", StringType()),
            StructField("title", StringType()),
            StructField("body", StringType()),
            StructField("author", StringType()),
            StructField("created_at", StringType()),
            StructField("score", IntegerType()),
            StructField("comments_count", IntegerType()),
        ])
    
    def read_kafka_stream(self, topic: str):
        """
        Read from Kafka topic
        
        Args:
            topic: Kafka topic name
            
        Returns:
            Spark DataFrame with streaming data
        """
        return self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.brokers) \
            .option("subscribe", topic) \
            .option("startingOffsets", "latest") \
            .load()
    
    def process_stream(self, df):
        """
        Process streaming data
        
        Args:
            df: Input streaming DataFrame
            
        Returns:
            Processed DataFrame
        """
        schema = self.create_schema()
        
        processed = df.select(
            from_json(col("value").cast("string"), schema).alias("data")
        ).select("data.*")
        
        return processed
    
    def aggregate_by_source(self, df):
        """
        Aggregate statistics by source
        
        Args:
            df: Input DataFrame
            
        Returns:
            Aggregated DataFrame
        """
        return df.groupBy("source") \
            .count() \
            .alias("total_posts")
    
    def write_to_console(self, query, checkpoint_location: str):
        """
        Write stream output to console for debugging
        
        Args:
            query: Streaming query
            checkpoint_location: Directory for checkpoint
            
        Returns:
            Streaming query object
        """
        return query.writeStream \
            .format("console") \
            .option("checkpointLocation", checkpoint_location) \
            .outputMode("append") \
            .start()
    
    def stop(self):
        """Stop the Spark session"""
        self.spark.stop()
