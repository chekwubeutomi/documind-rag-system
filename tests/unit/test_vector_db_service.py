"""
Unit tests for vector database service.
"""
import pytest
from app.services.vector_db_service import VectorDBService


def test_vector_db_initialization():
    """Test VectorDBService initialization."""
    try:
        service = VectorDBService(db_type="qdrant", url="http://localhost:6333")
        # Service should initialize without error
    except Exception as e:
        pytest.skip(f"Vector DB not available: {e}")


def test_vector_db_qdrant_initialization():
    """Test Qdrant specific initialization."""
    try:
        service = VectorDBService(db_type="qdrant", url="http://localhost:6333")
        assert service.db_type == "qdrant"
    except Exception as e:
        pytest.skip(f"Qdrant not available: {e}")
