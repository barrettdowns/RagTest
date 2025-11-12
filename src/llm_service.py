"""
LLM Service Module
- Interfaces with OpenAI for generating responses
"""
import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
from src.utils.secrets import get_secret

# Load environment variables
load_dotenv()

class LLMService:
    def __init__(self):
        """
        Initialize the LLM service with OpenAI configuration.
        """
        api_key = get_secret("OPENAI_API_KEY")
        model_name = get_secret("OPENAI_MODEL", "gpt-3.5-turbo")
        
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please set it in Streamlit secrets or environment variables.\n"
                "For Streamlit Cloud: Go to Settings > Secrets and add:\n"
                "OPENAI_API_KEY=your-api-key-here"
            )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # Get model name from environment
        self.model_name = model_name
        
        if not self.model_name:
            raise ValueError("OPENAI_MODEL environment variable not set")
    
    def generate_response(
        self, 
        query: str, 
        context_documents: List[Dict[str, Any]],
        extracted_entities: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response to a query using retrieved context documents.
        
        Args:
            query: The user's query
            context_documents: List of relevant documents for context
            extracted_entities: Optional list of extracted entities
            
        Returns:
            Dictionary with the response, reasoning, and entities
        """
        # Prepare the context from retrieved documents
        context = self._prepare_context(context_documents)
        
        # Build the system message with instructions
        system_message = self._build_system_message(context)
        
        # Build the user message with the query
        user_message = self._build_user_message(query)
        
        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Extract the response content
        response_content = response.choices[0].message.content
        
        try:
            # Parse the JSON response
            response_json = json.loads(response_content)
            
            # Add extracted entities if available
            if extracted_entities and "entities" not in response_json:
                response_json["entities"] = extracted_entities
            
            return response_json
        except json.JSONDecodeError:
            # Fallback if the response is not valid JSON
            return {
                "answer": "Error: Could not parse LLM response as JSON",
                "reasoning": "The LLM did not return valid JSON. Original response: " + response_content[:100] + "...",
                "entities": extracted_entities or []
            }
    
    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Prepare the context from retrieved documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, doc in enumerate(documents):
            source = doc["metadata"].get("source", "Unknown")
            context_parts.append(f"Document {i+1} (Source: {source}):\n{doc['text']}")
        
        return "\n\n".join(context_parts)
    
    def _build_system_message(self, context: str) -> str:
        """
        Build the system message with instructions and context.
        
        Args:
            context: The document context
            
        Returns:
            System message string
        """
        return f"""You are an AI assistant that answers questions based on the provided documents.
Your responses must be in JSON format with the following structure:
{{
  "answer": "Your concise and informative answer",
  "reasoning": "Your step-by-step reasoning explaining how you arrived at the answer (optional)"
}}

If you're uncertain or if the information is ambiguous, note this in your reasoning.
Base your answers primarily on the provided context documents, but you can use your general knowledge for clarification.

### Context Documents:
{context}

### Instructions:
1. Answer questions using ONLY the provided documents and your general knowledge when necessary.
2. Include specific document references in your reasoning.
3. If the documents contain conflicting information, note this and explain your conclusion.
4. If the documents don't contain the answer, state that clearly.
5. Your response must be a JSON object with "answer" and "reasoning" fields.
"""
    
    def _build_user_message(self, query: str) -> str:
        """
        Build the user message with the query.
        
        Args:
            query: The user's query
            
        Returns:
            User message string
        """
        return f"""Please answer the following question based on the context documents:

{query}

Provide your answer in JSON format with "answer" and "reasoning" fields."""