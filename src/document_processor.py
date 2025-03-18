"""
Document Processing Module
- Parses PDF and text documents
- Splits documents into chunks suitable for embedding
"""
import os
import re
from typing import List, Dict, Union, Optional
import pypdf
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import streamlit as st

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: The target size of each text chunk (in tokens)
            chunk_overlap: The overlap between chunks (in tokens)
        """
        # Get chunk settings from Streamlit secrets or environment variables
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def process_file(self, file_path: str) -> List[Document]:
        """
        Process a file and return a list of document chunks.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of Document objects containing text chunks with metadata
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            text = self._parse_pdf(file_path)
        elif file_extension in ['.txt', '.md', '.html']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Extract filename for metadata
        filename = os.path.basename(file_path)
        
        # Split the text into chunks
        return self._split_text(text, metadata={"source": filename})
    
    def process_text(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        """
        Process raw text and return a list of document chunks.
        
        Args:
            text: The text content to process
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of Document objects containing text chunks with metadata
        """
        if metadata is None:
            metadata = {"source": "direct_text_input"}
            
        return self._split_text(text, metadata)
    
    def _parse_pdf(self, file_path: str) -> str:
        """
        Parse a PDF file and extract its text content.
        Try multiple methods for better extraction quality.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        text = ""
        
        # First attempt: pdfplumber (better for formatted PDFs)
        try:
            with pdfplumber.open(file_path) as pdf:
                pages_text = [page.extract_text() or "" for page in pdf.pages]
                plumber_text = "\n".join(pages_text)
                if plumber_text.strip():
                    text = plumber_text
        except Exception as e:
            print(f"Error extracting text with pdfplumber: {e}")
        
        # Second attempt: pypdf (if pdfplumber failed or returned empty text)
        if not text.strip():
            try:
                pdf_text = []
                with open(file_path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        content = page.extract_text()
                        if content:
                            pdf_text.append(content)
                text = "\n".join(pdf_text)
            except Exception as e:
                print(f"Error extracting text with pypdf: {e}")
        
        # Clean up the text
        text = self._clean_text(text)
        
        if not text.strip():
            raise ValueError(f"Could not extract text from PDF: {file_path}")
            
        return text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and fixing common issues.
        
        Args:
            text: The raw extracted text
            
        Returns:
            Cleaned text
        """
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def _split_text(self, text: str, metadata: Dict) -> List[Document]:
        """
        Split text into chunks for embedding.
        
        Args:
            text: The text to split
            metadata: Metadata to include with each chunk
            
        Returns:
            List of Document objects
        """
        docs = []
        chunks = self.text_splitter.split_text(text)
        
        for i, chunk in enumerate(chunks):
            # Create a unique ID for each chunk
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            
            # Create a Document object
            doc = Document(page_content=chunk, metadata=chunk_metadata)
            docs.append(doc)
        
        return docs