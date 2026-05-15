#!/bin/bash

set -e

echo "=== Social Media Storage System Startup ==="
echo ""

# Check if Docker is running
echo "Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running"
    exit 1
fi

# Start Docker services
echo "Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Initialize databases
echo "Initializing databases..."
python scripts/init_databases.py

echo ""
echo "=== System is ready! ==="
echo ""
echo "Services running:"
echo "  - Kafka:     localhost:9092"
echo "  - MongoDB:   localhost:27017"
echo "  - Cassandra: localhost:9042"
echo ""
echo "To stop the system, run: docker-compose down"
