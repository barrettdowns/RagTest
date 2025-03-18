"""
Embedding Engine Module
- Generates embeddings for text chunks using Azure OpenAI
"""
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain.docstore.document import Document
import streamlit as st

# Load environment variables
load_dotenv()

class EmbeddingEngine:
    def __init__(self):
        """
        Initialize the embedding engine with Azure OpenAI configuration.
        """
        try:
            # Try to access Streamlit secrets
            has_secrets = hasattr(st, "secrets")
            azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"] if has_secrets and "AZURE_OPENAI_ENDPOINT" in st.secrets else os.environ.get("AZURE_OPENAI_ENDPOINT")
            azure_deployment = st.secrets["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"] if has_secrets and "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME" in st.secrets else os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
            api_key = st.secrets["AZURE_OPENAI_API_KEY"] if has_secrets and "AZURE_OPENAI_API_KEY" in st.secrets else os.environ.get("AZURE_OPENAI_API_KEY")
            api_version = st.secrets["AZURE_OPENAI_API_VERSION"] if has_secrets and "AZURE_OPENAI_API_VERSION" in st.secrets else os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
        except Exception:
            # Fallback to environment variables
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            azure_deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")

        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            openai_api_key=api_key,
            openai_api_version=api_version,
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