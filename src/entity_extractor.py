"""
Entity Extractor Module (Optional)
- Uses Hugging Face models to extract named entities
"""
import os
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import streamlit as st
from src.utils.secrets import get_secret

# Force NumPy to be accessible
try:
    import numpy
    # Add NumPy's path to sys.path if it's not there
    numpy_path = os.path.dirname(numpy.__file__)
    if numpy_path not in sys.path:
        sys.path.append(numpy_path)
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy is not available. Entity extraction will be disabled.")

# Try importing torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch is not available. Entity extraction will be disabled.")

# Conditional imports for transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers library not available. Entity extraction will be disabled.")
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Warning: Error importing transformers: {e}. Entity extraction will be disabled.")

# Load environment variables
load_dotenv()

class EntityExtractor:
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """
        Initialize the entity extractor with a Hugging Face model.
        
        Args:
            model_name: Name of the Hugging Face model to use
        """
        self.enabled = TRANSFORMERS_AVAILABLE and NUMPY_AVAILABLE and TORCH_AVAILABLE
        self.model_name = model_name
        self.ner_pipeline = None
        
        if not self.enabled:
            print("Entity extraction disabled due to missing dependencies.")
            return
        
        try:
            # Get Hugging Face token
            hf_token = get_secret("HUGGINGFACE_API_TOKEN")
            
            # Create a custom tokenizer with explicit device assignment
            tokenizer_kwargs = {}
            if hf_token:
                tokenizer_kwargs["token"] = hf_token
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
            
            # Use CPU device explicitly to avoid MPS issues
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model_name,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device="cpu"  # Force CPU usage
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
            # Return empty list but don't disable the extractor for future attempts
            return []

# Create a dummy extractor
class DummyEntityExtractor:
    def __init__(self, *args, **kwargs):
        self.enabled = False
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        return []

# Determine which extractor to use
if TRANSFORMERS_AVAILABLE and NUMPY_AVAILABLE and TORCH_AVAILABLE:
    DefaultEntityExtractor = EntityExtractor
else:
    DefaultEntityExtractor = DummyEntityExtractor