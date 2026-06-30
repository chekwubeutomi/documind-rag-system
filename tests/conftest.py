"""
Pytest fixtures and configuration for tests.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What is FastAPI?"


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.6+.
    It is based on Starlette for the web parts and Pydantic for the data parts.
    FastAPI comes with automatic interactive API documentation."""


@pytest.fixture
def sample_chunks():
    """Sample text chunks."""
    return [
        "FastAPI is a modern, fast web framework for building APIs",
        "It is based on Starlette and Pydantic",
        "FastAPI comes with automatic interactive documentation"
    ]
