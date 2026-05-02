import json
import time
import random
from datetime import datetime

# from kafka import KafkaProducer 

def generate_transaction():
    brands = ['Apple', 'Samsung', 'Xiaomi', 'Sony', 'LG', 'Asus', 'Dell']
    categories = ['Smartphone', 'Laptop', 'Smartwatch', 'TV', 'Tablet']
    
    return {
        'transaction_id': str(random.randint(100000, 999999)),
        'customer_id': random.randint(100, 999),
        'brand': random.choice(brands),
        'category': random.choice(categories),
        'amount': round(random.uniform(50.0, 3000.0), 2),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

if __name__ == "__main__":
    print("--- Đang bắt đầu tạo dữ liệu giả lập (Streaming) ---")
    while True:
        data = generate_transaction()
        # In ra màn hình để Linh kiểm tra dữ liệu có chạy không
        print(f"Đã tạo giao dịch: {data}")
        time.sleep(1) # Tạo 1 giao dịch mỗi giây