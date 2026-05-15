"""Streaming module for Kafka and Spark processing"""

from .kafka_producer import KafkaProducer
from .spark_processor import SparkProcessor

__all__ = ['KafkaProducer', 'SparkProcessor']
