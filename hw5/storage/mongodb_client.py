"""MongoDB client for document storage"""

import os
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from dotenv import load_dotenv

load_dotenv()


class MongoDBClient:
    """MongoDB client for storing and querying social media data"""
    
    def __init__(self):
        host = os.getenv("MONGO_HOST", "localhost")
        port = int(os.getenv("MONGO_PORT", 27017))
        username = os.getenv("MONGO_USERNAME", "admin")
        password = os.getenv("MONGO_PASSWORD", "password123")
        database = os.getenv("MONGO_DATABASE", "social_media")
        
        try:
            connection_string = f"mongodb://{username}:{password}@{host}:{port}/"
            self.client = MongoClient(connection_string)
            self.db = self.client[database]
            self.client.admin.command("ping")
            print(f"Connected to MongoDB at {host}:{port}")
        except ConnectionFailure as e:
            print(f"Error connecting to MongoDB: {e}")
            self.client = None
            self.db = None
    
    def insert_one(self, collection: str, document: Dict[str, Any]) -> Optional[str]:
        """
        Insert a single document
        
        Args:
            collection: Collection name
            document: Document to insert
            
        Returns:
            Inserted document ID
        """
        try:
            result = self.db[collection].insert_one(document)
            return str(result.inserted_id)
        except OperationFailure as e:
            print(f"Error inserting document: {e}")
            return None
    
    def insert_many(self, collection: str, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Insert multiple documents
        
        Args:
            collection: Collection name
            documents: List of documents to insert
            
        Returns:
            List of inserted document IDs
        """
        try:
            result = self.db[collection].insert_many(documents, ordered=False)
            return [str(id) for id in result.inserted_ids]
        except OperationFailure as e:
            print(f"Error inserting documents: {e}")
            return []
    
    def find(self, collection: str, query: Dict[str, Any] = None, limit: int = 0) -> List[Dict]:
        """
        Find documents matching query
        
        Args:
            collection: Collection name
            query: Query filter
            limit: Maximum number of documents (0 = unlimited)
            
        Returns:
            List of matching documents
        """
        try:
            cursor = self.db[collection].find(query or {})
            if limit > 0:
                cursor = cursor.limit(limit)
            return list(cursor)
        except OperationFailure as e:
            print(f"Error querying documents: {e}")
            return []
    
    def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict]:
        """
        Find a single document
        
        Args:
            collection: Collection name
            query: Query filter
            
        Returns:
            Matching document or None
        """
        try:
            return self.db[collection].find_one(query)
        except OperationFailure as e:
            print(f"Error finding document: {e}")
            return None
    
    def update_one(self, collection: str, query: Dict[str, Any], update: Dict[str, Any]) -> bool:
        """
        Update a single document
        
        Args:
            collection: Collection name
            query: Query filter
            update: Update operations
            
        Returns:
            True if document was updated
        """
        try:
            result = self.db[collection].update_one(query, {"$set": update})
            return result.modified_count > 0
        except OperationFailure as e:
            print(f"Error updating document: {e}")
            return False
    
    def delete_one(self, collection: str, query: Dict[str, Any]) -> bool:
        """
        Delete a single document
        
        Args:
            collection: Collection name
            query: Query filter
            
        Returns:
            True if document was deleted
        """
        try:
            result = self.db[collection].delete_one(query)
            return result.deleted_count > 0
        except OperationFailure as e:
            print(f"Error deleting document: {e}")
            return False
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
