"""
Integration tests for API endpoints.
"""
import pytest


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "services" in data


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_query_empty_query(client):
    """Test query endpoint with empty query."""
    response = client.post(
        "/api/v1/query",
        json={"query": ""}
    )
    
    assert response.status_code == 400
