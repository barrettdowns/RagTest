"""
Vector Store Module
- Interfaces with ChromaDB for storing and retrieving embeddings
"""
import os
import chromadb
from typing import List, Dict, Any, Optional
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class VectorStore:
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store with ChromaDB.
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist ChromaDB data
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create the persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Using existing collection: {self.collection_name}")
        except Exception as e:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(name=self.collection_name)
            print(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with id, text, metadata, and embedding
        """
        if not documents:
            return
        
        # Extract required fields for ChromaDB
        ids = [doc["id"] for doc in documents]
        embeddings = [doc["embedding"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"Added {len(documents)} documents to collection {self.collection_name}")
    
    def query(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: The embedding vector of the query
            n_results: Number of results to return
            
        Returns:
            List of documents most similar to the query
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results for easier consumption
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results.get("distances", [[]])[0][i] if "distances" in results else None
            })
        
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "collection_name": self.collection_name,
            "count": self.collection.count()
        }
    
    def reset(self) -> None:
        """
        Reset the collection by deleting and recreating it.
        """
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name)
        print(f"Reset collection: {self.collection_name}")