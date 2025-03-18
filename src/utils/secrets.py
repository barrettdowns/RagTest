"""
Helper utilities for accessing secrets and environment variables
"""
import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_secret(key, default=None):
    """
    Get a value from Streamlit secrets or environment variables without printing errors.
    
    Args:
        key: The key to look for
        default: Default value if key is not found
        
    Returns:
        The value or default
    """
    try:
        # Try getting from Streamlit secrets
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        # Silently fail
        pass
    
    # Fall back to environment variables
    return os.environ.get(key, default)
