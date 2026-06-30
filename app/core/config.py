"""
Application configuration and settings.

This module provides centralized configuration management for the DocMind RAG system.
It loads configuration values from environment variables with sensible defaults,
making it easy to deploy across different environments (dev, staging, production).

All settings are defined in the Settings class which uses Pydantic BaseSettings
for automatic validation and type coercion. The .env file is loaded automatically.
"""
import os
from typing import Optional
#from pydantic import BaseSettings
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application configuration settings loaded from environment variables.
    
    This class uses Pydantic's BaseSettings to provide type-safe configuration
    with automatic validation. All values can be overridden via environment
    variables or .env file. Default values are provided as fallbacks.
    
    How it works:
    1. Pydantic looks for environment variables matching the field names
    2. Falls back to the default value provided in the field definition
    3. Validates and converts types (e.g., "True" string -> bool)
    4. Makes configuration available globally via the 'settings' instance
    
    Example usage in code:
        from app.core.config import settings
        model_name = settings.EMBEDDING_MODEL  # Returns "all-MiniLM-L6-v2" by default
    """
    
    # ============================================================================
    # APPLICATION METADATA - General app information and versioning
    # ============================================================================
    APP_NAME: str = "DocMind RAG"  # Display name of the application
    APP_VERSION: str = "1.0.0"  # Current version (follows semantic versioning)
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"  # Debug mode flag
    
    # ============================================================================
    # API SETTINGS - RESTful API configuration
    # ============================================================================
    API_V1_STR: str = "/api/v1"  # Base path for API v1 endpoints (e.g., /api/v1/query)
    
    # ============================================================================
    # VECTOR DATABASE CONFIGURATION
    # Manages where and how embeddings are stored and retrieved
    # ============================================================================
    VECTOR_DB_URL: str = os.getenv("VECTOR_DB_URL", "http://localhost:6333")
    # URL to access the vector database (Qdrant runs on port 6333 by default)
    
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "qdrant")
    # Type of vector database to use: "qdrant" (recommended) or "chroma"
    # Qdrant: Fast, scalable, production-ready vector search engine
    # ChromaDB: Lightweight, good for development and small deployments
    
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "documents")
    # Name of the collection in Qdrant where document embeddings are stored
    # Collections are like tables in traditional databases
    
    QDRANT_VECTOR_SIZE: int = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))
    # Dimension of embedding vectors (must match the embedding model)
    # all-MiniLM-L6-v2 produces 384-dimensional vectors
    
    # ============================================================================
    # EMBEDDING MODEL CONFIGURATION
    # Settings for the semantic embedding model (converts text to vectors)
    # ============================================================================
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    # Which embedding model to use from HuggingFace/Sentence-Transformers
    # all-MiniLM-L6-v2: Lightweight, fast, good for general purpose
    # Other options: all-mpnet-base-v2 (better quality, larger), nomic-embed-text
    
    EMBEDDING_MODEL_CACHE: str = os.getenv("EMBEDDING_MODEL_CACHE", "./models/sentence_transformers")
    # Local directory where downloaded embedding models are cached
    # Prevents re-downloading the same model on every restart
    
    # ============================================================================
    # LLM PROVIDER CONFIGURATION
    # Settings for large language models used to generate answers
    # ============================================================================
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    # Which LLM service to use for answer generation
    # Options: "groq" (default, fastest), "ollama" (local), "openai" (most capable)
    
    # Groq Configuration - Fast, free/cheap, optimized inference
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    # API key for Groq service (get from https://console.groq.com/)
    # Note: Can be None if not using Groq provider
    
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
    # Which Groq model to use. Mixtral-8x7b is fast and capable
    
    # Ollama Configuration - Run LLM locally (privacy-friendly)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # URL where local Ollama server is running (default port 11434)
    # Start Ollama with: ollama serve
    
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
    # Which Ollama model to use (must be pulled first: ollama pull mistral)
    
    # OpenAI Configuration - Most capable but requires API key and payment
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    # API key for OpenAI (get from https://platform.openai.com/api-keys)
    
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    # Which OpenAI model to use. gpt-4-turbo for better quality
    
    # ============================================================================
    # DOCUMENT PROCESSING CONFIGURATION
    # Settings that control how documents are split into chunks for embedding
    # ============================================================================
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    # Size of each text chunk in characters
    # 512 chars ≈ 100-150 words, good balance between context and granularity
    # Larger = more context, smaller = more retrieval results
    
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    # Number of characters overlapping between consecutive chunks
    # Ensures that important information near chunk boundaries isn't lost
    # Example: if CHUNK_SIZE=512 and OVERLAP=50, chunks share 50 characters
    
    # ============================================================================
    # RAG RETRIEVAL CONFIGURATION
    # Controls the retrieval-augmented generation pipeline behavior
    # ============================================================================
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    # Number of most similar document chunks to retrieve from vector database
    # Higher = more context but slower; lower = faster but may miss info
    # Typically 3-10 is optimal
    
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    # Minimum similarity score (0-1) for retrieved documents to be included
    # 0.5 = moderate relevance. Set higher to be more strict, lower to be permissive
    # Score < threshold = document is ignored
    
    # ============================================================================
    # MONITORING CONFIGURATION
    # Settings for observability and performance metrics
    # ============================================================================
    ENABLE_PROMETHEUS: bool = os.getenv("ENABLE_PROMETHEUS", "True").lower() == "true"
    # Enable Prometheus metrics collection for monitoring
    # Metrics available at http://localhost:8000/metrics
    
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "8001"))
    # Port where Prometheus metrics are exposed
    
    # ============================================================================
    # LOGGING CONFIGURATION
    # Settings that control logging behavior and verbosity
    # ============================================================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    # Minimum log level to capture: DEBUG, INFO, WARNING, ERROR, CRITICAL
    # DEBUG = verbose, shows everything; INFO = normal; WARNING+ = minimal
    
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/app.log")
    # File path where logs are written (in addition to console output)
    # Directory will be created automatically
    
    # ============================================================================
    # FILE UPLOAD CONFIGURATION
    # Settings that control document upload and storage
    # ============================================================================
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", "52428800"))
    # Maximum file upload size in bytes (default: 50MB = 52428800 bytes)
    # Files larger than this are rejected for security and performance
    
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./data/raw")
    # Directory where uploaded documents are temporarily stored
    # Should have sufficient disk space for your documents
    
    # ============================================================================
    # PYDANTIC CONFIGURATION
    # Internal Pydantic settings for the Settings class itself
    # ============================================================================
    class Config:
        env_file = ".env"  # Load configuration from .env file
        case_sensitive = True  # Environment variables are case-sensitive (all caps)


# ============================================================================
# GLOBAL SETTINGS INSTANCE
# Create a single Settings instance that's imported throughout the app
# ============================================================================
# This instance is used globally by all modules:
#   from app.core.config import settings
#   print(settings.CHUNK_SIZE)  # Access any configuration value
settings = Settings()
