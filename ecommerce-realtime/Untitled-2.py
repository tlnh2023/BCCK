# producer/ecommerce_producer.py

import json
import time
import random
import uuid
from datetime import datetime
from kafka import KafkaProducer

# Cấu hình Kafka
KAFKA_CONFIG = {
    'bootstrap_servers': 'localhost:9092',
    'topic': 'ecommerce-transactions',
    'acks': 'all',           # Đảm bảo ghi thành công trên tất cả replica
    'compression_type': 'gzip'  # Nén dữ liệu để tiết kiệm băng thông
}

# Danh mục sản phẩm với trọng số (weight)
PRODUCT_CATALOG = [
    # Apple - 40% tổng sản phẩm
    {"brand": "Apple", "category": "Phone", "product": "iPhone 15", "price": 999, "weight": 20},
    {"brand": "Apple", "category": "Phone", "product": "iPhone 15 Pro", "price": 1199, "weight": 15},
    {"brand": "Apple", "category": "Laptop", "product": "MacBook Air M3", "price": 1199, "weight": 10},
    {"brand": "Apple", "category": "Laptop", "product": "MacBook Pro 14\"", "price": 1999, "weight": 8},
    {"brand": "Apple", "category": "Audio", "product": "AirPods Pro", "price": 249, "weight": 15},
    
    # Samsung - 30%
    {"brand": "Samsung", "category": "Phone", "product": "Galaxy S24", "price": 849, "weight": 18},
    {"brand": "Samsung", "category": "Phone", "product": "Galaxy Z Fold5", "price": 1799, "weight": 5},
    {"brand": "Samsung", "category": "TV", "product": "Neo QLED 4K", "price": 1299, "weight": 10},
    {"brand": "Samsung", "category": "Audio", "product": "Galaxy Buds2", "price": 149, "weight": 15},
    
    # Xiaomi - 15%
    {"brand": "Xiaomi", "category": "Phone", "product": "Xiaomi 14", "price": 499, "weight": 15},
    {"brand": "Xiaomi", "category": "Phone", "product": "Xiaomi 14 Ultra", "price": 799, "weight": 8},
    {"brand": "Xiaomi", "category": "Smart Home", "product": "Robot Vacuum S5", "price": 399, "weight": 5},
    {"brand": "Xiaomi", "category": "Audio", "product": "Mi True Wireless", "price": 69, "weight": 12},
    
    # Sony - 8%
    {"brand": "Sony", "category": "Audio", "product": "WH-1000XM5", "price": 399, "weight": 10},
    {"brand": "Sony", "category": "Gaming", "product": "PlayStation 5", "price": 499, "weight": 6},
    {"brand": "Sony", "category": "Camera", "product": "Alpha 7 IV", "price": 2499, "weight": 3},
    
    # LG - 7%
    {"brand": "LG", "category": "TV", "product": "OLED evo C3", "price": 1599, "weight": 8},
    {"brand": "LG", "category": "Appliances", "product": "Washing Machine", "price": 899, "weight": 5},
]

def generate_transaction():
    """Sinh một giao dịch ngẫu nhiên dựa trên trọng số"""
    # Chọn sản phẩm theo trọng số
    product = random.choices(
        PRODUCT_CATALOG, 
        weights=[p["weight"] for p in PRODUCT_CATALOG]
    )[0]
    
    quantity = random.randint(1, 3)
    total_amount = round(product["price"] * quantity, 2)
    
    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": f"user_{random.randint(1, 10000)}",
        "event_time": datetime.now().isoformat() + "Z",
        "brand": product["brand"],
        "category": product["category"],
        "product_name": product["product"],
        "price": product["price"],
        "quantity": quantity,
        "total_amount": total_amount
    }

def run_producer(speed_factor=1.0):
    """
    Chạy producer với tốc độ có thể điều chỉnh
    
    Args:
        speed_factor: Hệ số tốc độ (1.0 = ~1 giao dịch/giây)
    """
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks=KAFKA_CONFIG['acks'],
        compression_type=KAFKA_CONFIG['compression_type']
    )
    
    print(f"✅ Producer started. Sending to topic: {KAFKA_CONFIG['topic']}")
    print(f"📊 Simulating {len(PRODUCT_CATALOG)} products from 5 brands\n")
    
    count = 0
    try:
        while True:
            transaction = generate_transaction()
            producer.send(KAFKA_CONFIG['topic'], value=transaction)
            count += 1
            
            # In log mỗi 10 giao dịch
            if count % 10 == 0:
                print(f"[{count}] 📤 Last: {transaction['brand']:8} | "
                      f"{transaction['product_name']:20} | ${transaction['total_amount']:8.2f}")
            
            # Delay ngẫu nhiên (0.5-1.5 giây) / speed_factor
            time.sleep(random.uniform(0.5, 1.5) / speed_factor)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Producer stopped. Total messages: {count}")
    finally:
        producer.flush()
        producer.close()

if __name__ == "__main__":
    run_producer(speed_factor=1.5)