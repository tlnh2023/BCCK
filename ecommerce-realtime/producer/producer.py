#!/usr/bin/env python3
"""
Producer Service - Simulates real-time ecommerce orders
Sends data to Kafka topic 'ecommerce-orders'
"""

import json
import time
import random
import uuid
from datetime import datetime
from kafka import KafkaProducer
from faker import Faker

fake = Faker()

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = ['kafka:9092']  # or 'localhost:9092' for local run
TOPIC_NAME = 'ecommerce-orders'

# Product catalog
PRODUCTS = [
    {"product_id": "p001", "name": "Gaming Laptop", "category": "Electronics", "price": 999.99},
    {"product_id": "p002", "name": "Wireless Mouse", "category": "Electronics", "price": 29.99},
    {"product_id": "p003", "name": "Mechanical Keyboard", "category": "Electronics", "price": 149.99},
    {"product_id": "p004", "name": "4K Monitor", "category": "Electronics", "price": 399.99},
    {"product_id": "p005", "name": "USB-C Hub", "category": "Electronics", "price": 49.99},
    {"product_id": "p006", "name": "Running Shoes", "category": "Sports", "price": 89.99},
    {"product_id": "p007", "name": "Yoga Mat", "category": "Sports", "price": 39.99},
    {"product_id": "p008", "name": "Dumbbell Set", "category": "Sports", "price": 79.99},
    {"product_id": "p009", "name": "Protein Powder", "category": "Sports", "price": 54.99},
    {"product_id": "p010", "name": "Coffee Mug", "category": "Home", "price": 15.99},
    {"product_id": "p011", "name": "Desk Lamp", "category": "Home", "price": 45.99},
    {"product_id": "p012", "name": "Throw Pillow", "category": "Home", "price": 24.99},
    {"product_id": "p013", "name": "T-Shirt", "category": "Fashion", "price": 24.99},
    {"product_id": "p014", "name": "Jeans", "category": "Fashion", "price": 79.99},
    {"product_id": "p015", "name": "Hoodie", "category": "Fashion", "price": 59.99},
    {"product_id": "p016", "name": "Python Book", "category": "Books", "price": 49.99},
    {"product_id": "p017", "name": "Data Science Guide", "category": "Books", "price": 59.99},
]

PAYMENT_METHODS = ['credit_card', 'paypal', 'cash', 'crypto']
ORDER_STATUSES = ['pending', 'processing', 'completed', 'shipped', 'delivered']

def create_kafka_producer():
    """Create and return Kafka producer"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',  # Wait for all replicas
            retries=3,
            max_in_flight_requests_per_connection=1
        )
        print(f"✅ Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
        return producer
    except Exception as e:
        print(f"❌ Failed to connect to Kafka: {e}")
        return None

def generate_order():
    """Generate a random ecommerce order"""
    product = random.choice(PRODUCTS)
    quantity = random.randint(1, 5)
    total_amount = product["price"] * quantity
    
    order = {
        "order_id": str(uuid.uuid4()),
        "user_id": fake.user_name() + str(random.randint(1, 1000)),
        "user_email": fake.email(),
        "product_id": product["product_id"],
        "product_name": product["name"],
        "category": product["category"],
        "quantity": quantity,
        "price": product["price"],
        "total_amount": round(total_amount, 2),
        "payment_method": random.choice(PAYMENT_METHODS),
        "order_status": random.choice(ORDER_STATUSES),
        "order_timestamp": datetime.now().isoformat(),
        "shipping_address": fake.address(),
        "customer_city": fake.city(),
        "customer_country": fake.country()
    }
    return order

def send_order(producer, order):
    """Send order to Kafka topic"""
    try:
        future = producer.send(
            TOPIC_NAME,
            key=order["user_id"],
            value=order
        )
        # Wait for send confirmation
        record_metadata = future.get(timeout=10)
        print(f"📦 Sent order {order['order_id'][:8]}... | Partition: {record_metadata.partition} | Offset: {record_metadata.offset}")
        return True
    except Exception as e:
        print(f"❌ Failed to send order: {e}")
        return False

def run_producer(interval_seconds=2, burst_mode=False, burst_count=10):
    """
    Run the producer continuously
    
    Args:
        interval_seconds: Time between orders (normal mode)
        burst_mode: If True, send burst_count orders quickly
        burst_count: Number of orders in burst mode
    """
    producer = create_kafka_producer()
    if not producer:
        return
    
    print(f"🚀 Producer started. Sending to topic: {TOPIC_NAME}")
    print(f"📊 Mode: {'BURST' if burst_mode else 'CONTINUOUS'}")
    print("-" * 50)
    
    order_count = 0
    
    if burst_mode:
        print(f"🔥 Sending {burst_count} orders in burst mode...")
        for i in range(burst_count):
            order = generate_order()
            if send_order(producer, order):
                order_count += 1
            time.sleep(0.1)  # Small delay between bursts
        print(f"✅ Burst complete. Sent {order_count} orders.")
    else:
        # Continuous mode
        try:
            while True:
                order = generate_order()
                if send_order(producer, order):
                    order_count += 1
                
                # Print status every 10 orders
                if order_count % 10 == 0:
                    print(f"📈 Total orders sent: {order_count}")
                
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print(f"\n🛑 Producer stopped. Total orders sent: {order_count}")
    
    producer.flush()
    producer.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ecommerce Order Producer')
    parser.add_argument('--interval', type=float, default=2, help='Interval between orders in seconds')
    parser.add_argument('--burst', action='store_true', help='Run in burst mode')
    parser.add_argument('--count', type=int, default=10, help='Number of orders in burst mode')
    parser.add_argument('--once', action='store_true', help='Send one order and exit')
    
    args = parser.parse_args()
    
    if args.once:
        # Send single order
        producer = create_kafka_producer()
        if producer:
            order = generate_order()
            send_order(producer, order)
            producer.flush()
            producer.close()
    else:
        run_producer(
            interval_seconds=args.interval,
            burst_mode=args.burst,
            burst_count=args.count
        )
