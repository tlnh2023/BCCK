"""Database initialization script"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.mongodb_client import MongoDBClient
from storage.cassandra_client import CassandraClient


def init_mongodb():
    """Initialize MongoDB collections and indexes"""
    print("Initializing MongoDB...")
    
    client = MongoDBClient()
    
    if not client.db:
        print("Failed to connect to MongoDB")
        return False
    
    collections = ["github_issues", "reddit_posts", "combined_posts"]
    
    for collection in collections:
        try:
            if collection not in client.db.list_collection_names():
                client.db.create_collection(collection)
                print(f"Created collection: {collection}")
            
            client.db[collection].create_index("id", unique=True)
            client.db[collection].create_index("created_at")
            client.db[collection].create_index("source")
            print(f"Created indexes for: {collection}")
        except Exception as e:
            print(f"Error initializing collection {collection}: {e}")
    
    client.close()
    print("MongoDB initialization complete")
    return True


def init_cassandra():
    """Initialize Cassandra tables"""
    print("Initializing Cassandra...")
    
    client = CassandraClient()
    
    if not client.session:
        print("Failed to connect to Cassandra")
        return False
    
    try:
        query = f"""
            CREATE TABLE IF NOT EXISTS {client.keyspace}.posts (
                id TEXT,
                source TEXT,
                title TEXT,
                author TEXT,
                created_at TEXT,
                score INT,
                comments_count INT,
                PRIMARY KEY ((source), created_at, id)
            )
        """
        client.execute(query)
        print("Created Cassandra table: posts")
    except Exception as e:
        print(f"Error creating Cassandra table: {e}")
        return False
    
    client.close()
    print("Cassandra initialization complete")
    return True


def main():
    """Initialize all databases"""
    print("Starting database initialization...\n")
    
    mongodb_ok = init_mongodb()
    print()
    cassandra_ok = init_cassandra()
    
    print()
    if mongodb_ok and cassandra_ok:
        print("All databases initialized successfully!")
        return 0
    else:
        print("Database initialization encountered errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
