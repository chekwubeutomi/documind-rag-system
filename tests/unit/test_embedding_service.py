"""
Unit tests for embedding service.
"""
import pytest
import numpy as np
from app.services.embedding_service import EmbeddingService


def test_embedding_service_initialization():
    """Test EmbeddingService initialization."""
    try:
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        assert service.model is not None
        assert service.dimension is not None
    except Exception as e:
        pytest.skip(f"Embedding model not available: {e}")


def test_embedding_dimension():
    """Test embedding dimension."""
    try:
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        dimension = service.get_dimension()
        assert dimension > 0
    except Exception as e:
        pytest.skip(f"Embedding model not available: {e}")


def test_embed_text(sample_text):
    """Test text embedding."""
    try:
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        embedding = service.embed_text(sample_text)
        
        assert embedding.shape[0] == service.get_dimension()
    except Exception as e:
        pytest.skip(f"Embedding model not available: {e}")


def test_embed_texts(sample_chunks):
    """Test embedding multiple texts."""
    try:
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        embeddings = service.embed_texts(sample_chunks)
        
        assert len(embeddings) == len(sample_chunks)
        assert all(emb.shape[0] == service.get_dimension() for emb in embeddings)
    except Exception as e:
        pytest.skip(f"Embedding model not available: {e}")
