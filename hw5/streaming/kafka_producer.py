"""Kafka producer for publishing social media data"""

import json
import os
from typing import Dict, Any
from kafka import KafkaProducer as KafkaProducerBase
from kafka.errors import KafkaError
from dotenv import load_dotenv

load_dotenv()


class KafkaProducer:
    """Kafka producer for streaming social media data"""
    
    def __init__(self, topic: str = "social-media-events"):
        self.topic = topic
        brokers = os.getenv("KAFKA_BROKERS", "localhost:9092").split(",")
        
        try:
            self.producer = KafkaProducerBase(
                bootstrap_servers=brokers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                retries=3
            )
        except Exception as e:
            print(f"Error initializing Kafka producer: {e}")
            self.producer = None
    
    def send(self, data: Dict[str, Any], key: str = None) -> bool:
        """
        Send data to Kafka topic
        
        Args:
            data: Data dictionary to send
            key: Message key for partitioning
            
        Returns:
            True if successful, False otherwise
        """
        if not self.producer:
            print("Producer not initialized")
            return False
        
        try:
            future = self.producer.send(
                self.topic,
                value=data,
                key=key.encode("utf-8") if key else None
            )
            future.get(timeout=10)
            return True
        except KafkaError as e:
            print(f"Error sending message to Kafka: {e}")
            return False
    
    def flush(self):
        """Flush pending messages"""
        if self.producer:
            self.producer.flush()
    
    def close(self):
        """Close the producer connection"""
        if self.producer:
            self.producer.close()
