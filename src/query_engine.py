"""
Query Engine Module
- Orchestrates the query process from user input to structured response
"""
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import json
import streamlit as st

from .document_processor import DocumentProcessor
from .embedding_engine import EmbeddingEngine
from .vector_store import VectorStore
from .llm_service import LLMService
from .entity_extractor import DefaultEntityExtractor

# Load environment variables
load_dotenv()

class QueryEngine:
    def __init__(self, use_entity_extraction: bool = False):
        """
        Initialize the query engine with its components.
        
        Args:
            use_entity_extraction: Whether to use entity extraction
        """
        # Get chunk settings from environment or secrets
        try:
            # Try Streamlit secrets first
            if hasattr(st, "secrets") and "CHUNK_SIZE" in st.secrets:
                chunk_size = int(st.secrets["CHUNK_SIZE"])
            else:
                chunk_size = int(os.environ.get("CHUNK_SIZE", "500"))
                
            if hasattr(st, "secrets") and "CHUNK_OVERLAP" in st.secrets:
                chunk_overlap = int(st.secrets["CHUNK_OVERLAP"])
            else:
                chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "50"))
        except:
            # Fallback to defaults
            chunk_size = 500
            chunk_overlap = 50
            
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.llm_service = LLMService()
        
        # Initialize entity extractor if enabled
        self.use_entity_extraction = use_entity_extraction
        if self.use_entity_extraction:
            self.entity_extractor = DefaultEntityExtractor()
        else:
            self.entity_extractor = None
    
    def index_document(self, file_path: str, original_filename: str = None) -> Dict[str, Any]:
        """
        Process and index a document.
        
        Args:
            file_path: Path to the document
            original_filename: Original filename to use in metadata (if different from file_path)
            
        Returns:
            Dictionary with indexing statistics
        """
        try:
            # Process the document
            documents = self.document_processor.process_file(file_path, original_filename)
            
            # Generate embeddings
            embedded_documents = self.embedding_engine.embed_documents(documents)
            
            # Store in vector store
            self.vector_store.add_documents(embedded_documents)
            
            return {
                "status": "success",
                "document": os.path.basename(file_path),
                "chunks": len(documents),
                "total_documents": self.vector_store.get_stats()["count"]
            }
        except Exception as e:
            return {
                "status": "error",
                "document": os.path.basename(file_path),
                "error": str(e)
            }
    
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Process a query and return a structured response.
        
        Args:
            query_text: The user's query
            top_k: Number of documents to retrieve
            
        Returns:
            Structured response with answer, reasoning, and optional entities
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_engine.embed_query(query_text)
            
            # Retrieve relevant documents
            relevant_docs = self.vector_store.query(query_embedding, n_results=top_k)
            
            # Extract entities if enabled
            extracted_entities = []
            if self.use_entity_extraction and self.entity_extractor and self.entity_extractor.enabled:
                # Extract entities from the query
                query_entities = self.entity_extractor.extract_entities(query_text)
                
                # Extract entities from relevant documents
                doc_entities = []
                for doc in relevant_docs:
                    doc_entities.extend(self.entity_extractor.extract_entities(doc["text"]))
                
                # Combine and deduplicate entities
                all_entities = query_entities + doc_entities
                seen_entities = set()
                filtered_entities = []
                
                for entity in all_entities:
                    entity_key = (entity["type"], entity["text"])
                    if entity_key not in seen_entities:
                        seen_entities.add(entity_key)
                        # Remove start/end positions and simplify for output
                        filtered_entities.append({
                            "type": entity["type"],
                            "text": entity["text"],
                            "score": entity.get("score", 1.0)
                        })
                
                extracted_entities = filtered_entities
            
            # Generate response using LLM
            response = self.llm_service.generate_response(
                query_text,
                relevant_docs,
                extracted_entities
            )
            
            return response
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "reasoning": "An error occurred during query processing.",
                "entities": []
            }
    
    def reset_index(self) -> Dict[str, str]:
        """
        Reset the vector store.
        
        Returns:
            Status message
        """
        try:
            self.vector_store.reset()
            return {"status": "success", "message": "Vector store reset successfully"}
        except Exception as e:
            return {"status": "error", "message": f"Error resetting vector store: {str(e)}"}