"""
Pydantic request and response schemas for API endpoints.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ============= Upload/Document Schemas =============

class DocumentChunk(BaseModel):
    """Schema for a chunked document."""
    content: str = Field(..., description="The text content of the chunk")
    metadata: dict = Field(default_factory=dict, description="Metadata associated with the chunk")
    chunk_index: int = Field(..., description="Index of the chunk within the document")


class DocumentUploadResponse(BaseModel):
    """Response schema for document upload."""
    document_id: str = Field(..., description="Unique identifier for the uploaded document")
    filename: str = Field(..., description="Name of the uploaded file")
    num_chunks: int = Field(..., description="Number of chunks created from the document")
    status: str = Field(default="success", description="Status of the upload")
    message: str = Field(..., description="Additional message")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of upload")


# ============= Query/Retrieval Schemas =============

class RetrievedDocument(BaseModel):
    """Schema for a retrieved document chunk."""
    content: str = Field(..., description="The text content")
    score: float = Field(..., description="Similarity score (0-1)")
    document_id: str = Field(..., description="Source document ID")
    metadata: dict = Field(default_factory=dict, description="Associated metadata")


class QueryRequest(BaseModel):
    """Request schema for querying the RAG system."""
    query: str = Field(..., min_length=1, description="The user query")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    include_sources: bool = Field(default=True, description="Whether to include source documents")
    stream: bool = Field(default=False, description="Whether to stream the response")


class QueryResponse(BaseModel):
    """Response schema for RAG query."""
    query: str = Field(..., description="The original query")
    answer: str = Field(..., description="Generated answer from the LLM")
    sources: List[RetrievedDocument] = Field(default_factory=list, description="Retrieved source documents")
    model: str = Field(..., description="LLM model used")
    processing_time: float = Field(..., description="Time taken to process (seconds)")
    tokens_used: Optional[dict] = Field(default=None, description="Token usage if available")


# ============= Vector DB Schemas =============

class EmbeddingRequest(BaseModel):
    """Request schema for embedding generation."""
    texts: List[str] = Field(..., description="List of texts to embed")


class EmbeddingResponse(BaseModel):
    """Response schema for embeddings."""
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    dimension: int = Field(..., description="Dimension of embeddings")
    model: str = Field(..., description="Embedding model used")


# ============= Health Check Schemas =============

class HealthCheck(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field(..., description="Overall status")
    version: str = Field(..., description="Application version")
    services: dict = Field(default_factory=dict, description="Status of each service")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of check")


# ============= Error Schemas =============

class ErrorDetail(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the error occurred")
