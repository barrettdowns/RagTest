"""
Streamlit Frontend for RAG Document Query System
- Provides UI for document upload, indexing, and querying
"""
import os
import json
import tempfile
from typing import List, Dict, Any
import streamlit as st
from dotenv import load_dotenv

# Import custom modules
from src.query_engine import QueryEngine

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="RAG Document Query System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "query_engine" not in st.session_state:
    st.session_state.query_engine = QueryEngine(use_entity_extraction=False)

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "query_history" not in st.session_state:
    st.session_state.query_history = []

# Create data directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# Title and description
st.title("RAG Document Query System")
st.markdown("""
This system demonstrates Retrieval-Augmented Generation (RAG) using:
- Document processing and chunking
- Vector embeddings with Azure OpenAI
- ChromaDB for vector storage
- LLM-based reasoning for answer generation
""")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Entity extraction toggle
    use_entity_extraction = st.toggle("Enable Entity Extraction", value=False)
    if use_entity_extraction != getattr(st.session_state.query_engine, "use_entity_extraction", False):
        st.session_state.query_engine = QueryEngine(use_entity_extraction=use_entity_extraction)
        st.success("Updated entity extraction setting")
    
    # Reset button
    if st.button("Reset Vector Store"):
        result = st.session_state.query_engine.reset_index()
        if result["status"] == "success":
            st.session_state.indexed_files = []
            st.success(result["message"])
        else:
            st.error(result["message"])
    
    # Show indexed files
    st.subheader("Indexed Documents")
    if st.session_state.indexed_files:
        for file in st.session_state.indexed_files:
            st.text(f"✅ {file}")
    else:
        st.text("No documents indexed yet")

# Main content
tab1, tab2 = st.tabs(["Document Upload & Indexing", "Query Documents"])

# Document Upload & Indexing Tab
with tab1:
    st.header("Document Upload")
    
    # Set maximum number of documents - check Streamlit secrets first, then environment variables
    try:
        has_secrets = hasattr(st, "secrets")
        max_documents = int(st.secrets["MAX_DOCUMENTS"]) if has_secrets and "MAX_DOCUMENTS" in st.secrets else int(os.environ.get("MAX_DOCUMENTS", "2"))
    except Exception:
        max_documents = int(os.environ.get("MAX_DOCUMENTS", "2"))
    
    st.info(f"Maximum number of documents for this POC: {max_documents}")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload document(s)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        # Validate number of files
        if len(uploaded_files) > max_documents:
            st.warning(f"Too many documents. Only the first {max_documents} will be processed.")
            uploaded_files = uploaded_files[:max_documents]
        
        # Display uploaded files
        for file in uploaded_files:
            st.session_state.uploaded_files.append(file.name)
            st.text(f"📄 {file.name} - {file.size} bytes")
        
        # Index button
        if st.button("Index Documents"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                # Update progress
                progress = (i / len(uploaded_files))
                progress_bar.progress(progress)
                status_text.text(f"Processing {file.name}...")
                
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                    tmp_file.write(file.getbuffer())
                    file_path = tmp_file.name
                
                # Index the document
                result = st.session_state.query_engine.index_document(file_path)
                
                # Clean up temporary file
                os.unlink(file_path)
                
                # Display result
                if result["status"] == "success":
                    if file.name not in st.session_state.indexed_files:
                        st.session_state.indexed_files.append(file.name)
                    st.success(f"Successfully indexed {file.name} - {result['chunks']} chunks")
                else:
                    st.error(f"Error indexing {file.name}: {result.get('error', 'Unknown error')}")
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.text("All documents processed")
            
            # Summary
            st.info(f"Total documents in vector store: {result.get('total_documents', 'unknown')}")

# Query Documents Tab
with tab2:
    st.header("Query Documents")
    
    # Check if any documents have been indexed
    if not st.session_state.indexed_files:
        st.warning("No documents have been indexed yet. Please upload and index documents first.")
    else:
        # Query input
        query = st.text_input("Enter your query", key="query_input")
        
        # Submit button
        if st.button("Submit Query") and query:
            # Process the query
            with st.spinner("Processing query..."):
                response = st.session_state.query_engine.query(query)
                
                # Add to query history
                st.session_state.query_history.append({
                    "query": query,
                    "response": response
                })
            
            # Display the response
            st.subheader("Answer")
            st.write(response.get("answer", "No answer generated"))
            
            # Display reasoning if available
            if "reasoning" in response:
                with st.expander("Reasoning", expanded=True):
                    st.write(response["reasoning"])
            
            # Display entities if available
            if "entities" in response and response["entities"]:
                with st.expander("Extracted Entities", expanded=True):
                    entities_df = []
                    for entity in response["entities"]:
                        entities_df.append({
                            "Type": entity.get("type", "UNKNOWN"),
                            "Text": entity.get("text", ""),
                            "Confidence": f"{entity.get('score', 1.0):.2f}"
                        })
                    
                    st.dataframe(entities_df)
            
            # Display raw JSON
            with st.expander("Raw JSON Response", expanded=False):
                st.json(response)
        
        # Query history
        if st.session_state.query_history:
            st.subheader("Query History")
            for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"Query: {item['query']}", expanded=i == 0):
                    st.write("**Answer:**")
                    st.write(item["response"].get("answer", "No answer generated"))
                    
                    if "reasoning" in item["response"]:
                        st.write("**Reasoning:**")
                        st.write(item["response"].get("answer", "No answer generated"))