"""
Embedding Engine Module
- Generates embeddings for text chunks using OpenAI
"""
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
try:
    from langchain_core.documents import Document
except ImportError:
    # Fallback for older langchain versions
    from langchain.docstore.document import Document
import streamlit as st
from src.utils.secrets import get_secret

# Load environment variables
load_dotenv()

class EmbeddingEngine:
    def __init__(self):
        """
        Initialize the embedding engine with OpenAI configuration.
        """
        api_key = get_secret("OPENAI_API_KEY")
        embedding_model = get_secret("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please set it in Streamlit secrets or environment variables.\n"
                "For Streamlit Cloud: Go to Settings > Secrets and add:\n"
                "OPENAI_API_KEY=your-api-key-here"
            )

        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model=embedding_model,
        )
    
    def embed_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of Document objects to embed
            
        Returns:
            List of dictionaries with embeddings and metadata
        """
        if not documents:
            return []
        
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Combine embeddings with document content and metadata
        embedded_documents = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            embedded_documents.append({
                "id": f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_id', i)}",
                "text": doc.page_content,
                "metadata": doc.metadata,
                "embedding": embedding
            })
        
        return embedded_documents
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.
        
        Args:
            query: The query text to embed
            
        Returns:
            List of floats representing the query embedding
        """
        return self.embeddings.embed_query(query)