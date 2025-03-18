"""
Example usage of the RAG Document Query System
- Demonstrates how to use the system programmatically
"""
import os
import json
from dotenv import load_dotenv

from src.query_engine import QueryEngine

# Load environment variables
load_dotenv()

def main():
    # Initialize the query engine
    print("Initializing query engine...")
    query_engine = QueryEngine(use_entity_extraction=True)
    
    # Define file paths (update these to your actual files)
    document_paths = [
        "data/sample_technical_doc.pdf",
        "data/sample_user_manual.txt"
    ]
    
    # Index documents
    for doc_path in document_paths:
        if os.path.exists(doc_path):
            print(f"Indexing document: {doc_path}")
            result = query_engine.index_document(doc_path)
            print(json.dumps(result, indent=2))
        else:
            print(f"Document not found: {doc_path}")
    
    # Example queries to try
    example_queries = [
        "What are the main components of the system architecture?",
        "How does the document processing work?",
        "What are the key advantages of ChromaDB?",
        "Can you explain the difference between Azure OpenAI and Hugging Face approaches?",
        "What are the limitations of the current implementation?",
    ]
    
    # Run queries
    for query in example_queries:
        print("\n" + "="*80)
        print(f"Query: {query}")
        print("="*80)
        
        response = query_engine.query(query)
        
        print("\nAnswer:")
        print(response.get("answer", "No answer generated"))
        
        print("\nReasoning:")
        print(response.get("reasoning", "No reasoning provided"))
        
        if "entities" in response and response["entities"]:
            print("\nExtracted Entities:")
            for entity in response["entities"]:
                print(f"- {entity['type']}: {entity['text']} (score: {entity.get('score', 1.0):.2f})")
        
        print("\n")

if __name__ == "__main__":
    main()