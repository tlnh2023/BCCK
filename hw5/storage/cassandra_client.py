"""Cassandra client for time-series data storage"""

import os
from typing import List, Dict, Any, Optional
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.util import uuid_from_time
from dotenv import load_dotenv

load_dotenv()


class CassandraClient:
    """Cassandra client for storing time-series social media data"""
    
    def __init__(self):
        hosts = os.getenv("CASSANDRA_HOSTS", "localhost").split(",")
        keyspace = os.getenv("CASSANDRA_KEYSPACE", "social_media")
        
        try:
            cluster = Cluster(hosts)
            self.session = cluster.connect()
            self.keyspace = keyspace
            print(f"Connected to Cassandra at {hosts}")
            self._ensure_keyspace()
        except Exception as e:
            print(f"Error connecting to Cassandra: {e}")
            self.session = None
    
    def _ensure_keyspace(self):
        """Create keyspace if it doesn't exist"""
        try:
            self.session.execute(f"""
                CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
                WITH REPLICATION = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
            """)
            self.session.set_keyspace(self.keyspace)
        except Exception as e:
            print(f"Error creating keyspace: {e}")
    
    def execute(self, query: str, parameters: tuple = None) -> List[tuple]:
        """
        Execute a CQL query
        
        Args:
            query: CQL query string
            parameters: Query parameters
            
        Returns:
            Query results as list of tuples
        """
        try:
            if parameters:
                result = self.session.execute(query, parameters)
            else:
                result = self.session.execute(query)
            return list(result)
        except Exception as e:
            print(f"Error executing query: {e}")
            return []
    
    def insert_post(self, post_data: Dict[str, Any]) -> bool:
        """
        Insert a post into Cassandra
        
        Args:
            post_data: Post data dictionary
            
        Returns:
            True if successful
        """
        try:
            query = f"""
                INSERT INTO {self.keyspace}.posts
                (id, source, title, author, created_at, score, comments_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            self.session.execute(query, (
                post_data.get("id"),
                post_data.get("source"),
                post_data.get("title"),
                post_data.get("author"),
                post_data.get("created_at"),
                post_data.get("score", 0),
                post_data.get("comments_count", 0)
            ))
            return True
        except Exception as e:
            print(f"Error inserting post: {e}")
            return False
    
    def insert_batch(self, posts: List[Dict[str, Any]]) -> int:
        """
        Insert multiple posts in batch
        
        Args:
            posts: List of post dictionaries
            
        Returns:
            Number of successfully inserted posts
        """
        count = 0
        for post in posts:
            if self.insert_post(post):
                count += 1
        return count
    
    def get_posts_by_source(self, source: str, limit: int = 100) -> List[Dict]:
        """
        Get posts by source
        
        Args:
            source: Data source (github, reddit)
            limit: Maximum number of posts
            
        Returns:
            List of posts
        """
        try:
            query = f"""
                SELECT * FROM {self.keyspace}.posts
                WHERE source = ?
                LIMIT ?
            """
            results = self.session.execute(query, (source, limit))
            return [dict(row) for row in results]
        except Exception as e:
            print(f"Error querying posts: {e}")
            return []
    
    def close(self):
        """Close Cassandra connection"""
        if self.session:
            self.session.cluster.shutdown()
