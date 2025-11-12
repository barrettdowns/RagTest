RAG Document Query System (Proof of Concept)
This repository contains a proof-of-concept implementation of a Retrieval-Augmented Generation (RAG) system for processing, indexing, and querying technical documents.
Features

Document Processing: Ingest PDF or text documents and split them into appropriate chunks
Vector Embeddings: Generate vector embeddings using OpenAI
Semantic Search: Store and retrieve document chunks using ChromaDB
LLM-Based Reasoning: Answer complex questions spanning multiple documents using OpenAI
Structured JSON Responses: Return well-formatted JSON with answers, reasoning, and optional entities
Entity Extraction (Optional): Identify named entities using Hugging Face models
Streamlit Interface: Simple UI for document upload, indexing, and querying

System Architecture
Copy      ┌─────────────┐          ┌────────────────┐
      │  Documents  │          │ Streamlit Front│
      │ (PDF/TXT)   │          │     End        │
      └────┬────────┘          └──────┬─────────┘
           │                          │
           ▼                          ▼
    ┌────────────────┐      ┌────────────────────┐
    │ Document Parser│      │ Query/Response      │
    │  (Splitting &  │<---->│ Orchestrator        │
    │   Cleaning)    │      │ (Custom)            │
    └────────────────┘      └────────────────────┘
           ▼                         │
    ┌────────────────┐              │
    │ Embeddings via │<-------------┘
    │ OpenAI         │
    └────────────────┘
           ▼
 ┌────────────────┐
 │  ChromaDB      │
 │  (Local Vector │
 │  Storage)      │
 └────────────────┘
Installation

Clone the repository:
bashCopygit clone https://github.com/yourusername/rag-poc.git
cd rag-poc

Create a virtual environment (recommended):
bashCopypython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
bashCopypip install -r requirements.txt

Set up environment variables:

For local development, create a `.env` file:
```bash
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-3.5-turbo  # or gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

For Streamlit Cloud deployment:
1. Go to your app settings in Streamlit Cloud
2. Click on "Secrets" in the left sidebar
3. Add the following secrets:
```toml
OPENAI_API_KEY = "your-openai-api-key-here"
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
```

Usage

Start the Streamlit application:
bashCopystreamlit run app.py

Open your browser and navigate to http://localhost:8501
Upload up to 10 documents (PDF or text)
Click "Index Documents" to process and store embeddings
Navigate to the "Query Documents" tab
Enter your question and click "Submit Query"

Design Choices and Trade-Offs
Document Processing
The system uses both pypdf and pdfplumber for PDF extraction, with a fallback mechanism to ensure the best possible text extraction. For challenging PDFs, additional extraction methods could be integrated.
Vector Storage: ChromaDB
ChromaDB was chosen for this POC because:

It's lightweight and runs locally without requiring a separate server
It supports persistent storage out of the box
It has a simple Python API that's easy to integrate
It offers good performance for small to medium document collections

For a production system, alternatives like Pinecone, Weaviate, or Qdrant might be more suitable if scaling beyond a few documents.
OpenAI vs. Hugging Face
The system primarily uses OpenAI for two key reasons:

Performance: OpenAI provides state-of-the-art models like GPT-4 with strong reasoning capabilities
Ease of use: Simple API integration with excellent documentation and support

Hugging Face is used as an optional component for entity extraction because:

It offers specialized models for Named Entity Recognition (NER)
It can run locally, reducing API calls and latency for this specific task

Modular Architecture
The system is designed with modularity in mind:

Each component (document processor, embedding engine, etc.) is isolated
Components can be replaced or enhanced independently
The architecture supports potential extensions like caching or additional document types

Limitations and Future Enhancements

Document Limit: The POC is limited to 10 documents for simplicity
PDF Handling: Complex PDFs with tables, images, or unusual formatting may not be parsed optimally
Entity Extraction: The current implementation is basic and could be enhanced with domain-specific models
Caching: Implementing a response cache would improve performance for repeated queries
Fine-tuning: Custom model fine-tuning could improve domain-specific accuracy

Example Queries
Try asking questions like:

"What are the key components of the system?"
"How does the document processing pipeline work?"
"What are the main advantages of using ChromaDB for this application?"
"Can you compare the Azure OpenAI and Hugging Face approaches?"

License
MIT License