"""
Entity Extractor Module (Optional)
- Uses Hugging Face models to extract named entities
"""
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import torch
import streamlit as st

# Conditional imports to make this module optional
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Load environment variables
load_dotenv()

class EntityExtractor:
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """
        Initialize the entity extractor with a Hugging Face model.
        
        Args:
            model_name: Name of the Hugging Face model to use
        """
        self.enabled = TRANSFORMERS_AVAILABLE
        self.model_name = model_name
        self.ner_pipeline = None
        
        if not self.enabled:
            print("Transformers library not available. Entity extraction disabled.")
            return
        
        try:
            # Initialize the NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model_name,
                tokenizer=AutoTokenizer.from_pretrained(self.model_name),
                aggregation_strategy="simple"
            )
            print(f"Entity extractor initialized with model: {self.model_name}")
        except Exception as e:
            print(f"Error initializing entity extractor: {e}")
            self.enabled = False
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            List of dictionaries with entity information
        """
        if not self.enabled or not self.ner_pipeline:
            return []
        
        try:
            # Extract entities using the pipeline
            entities = self.ner_pipeline(text)
            
            # Format the results
            formatted_entities = []
            for entity in entities:
                formatted_entities.append({
                    "type": entity["entity_group"],
                    "text": entity["word"],
                    "score": float(entity["score"]),
                    "start": entity["start"],
                    "end": entity["end"]
                })
            
            return formatted_entities
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []

# Create a dummy extractor that returns empty results when Hugging Face is not available
class DummyEntityExtractor:
    def __init__(self, *args, **kwargs):
        self.enabled = False
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        return []

# Use the appropriate class based on availability
if TRANSFORMERS_AVAILABLE:
    DefaultEntityExtractor = EntityExtractor
else:
    DefaultEntityExtractor = DummyEntityExtractor