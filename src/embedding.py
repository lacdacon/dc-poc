#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables (in case this is imported before .env is loaded)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

def get_embedding_function():
    """
    Returns an OpenAI embeddings function for vectorizing text.
    
    Uses text-embedding-3-large model (3072 dimensions) for high-quality embeddings.
    Falls back to text-embedding-3-small if specified via environment variable.
    
    Returns:
        OpenAIEmbeddings: Configured embedding function
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please check your .env file."
        )
    
    # Allow model override via environment variable
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=api_key,
        # Optional: reduce dimensions for faster processing (uncomment if needed)
        # dimensions=1536,  # text-embedding-3-large supports 256-3072
    )