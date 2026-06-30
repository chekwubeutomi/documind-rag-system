"""
Pydantic request and response schemas for API endpoints.

This module defines all data models used by the FastAPI REST API. These schemas:
1. Validate incoming request data automatically (FastAPI + Pydantic)
2. Document API contracts in the automatically generated Swagger/OpenAPI docs
3. Provide type hints for IDE autocomplete and type checking
4. Serialize/deserialize data to/from JSON

How Pydantic Schemas Work:
- When a request is received, FastAPI uses the schema class to validate the JSON
- If validation fails, a 422 error is returned with details about what's wrong
- Response schemas ensure API responses have a consistent, documented format
- Field() provides metadata like descriptions shown in API docs

Example:
    POST /api/v1/query with JSON {"query": "test", "top_k": 5}
    FastAPI validates using QueryRequest schema
    API endpoint receives validated data with correct types
    Endpoint returns QueryResponse which is automatically serialized to JSON
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# DOCUMENT UPLOAD & PROCESSING SCHEMAS
# Used when uploading documents and processing them into chunks
# ============================================================================

class DocumentChunk(BaseModel):
    """
    Schema representing a single chunk of a larger document.
    
    When documents are processed, they're split into smaller chunks for
    embedding and storage. Each chunk is a portion of text with metadata
    about its position and source.
    
    Attributes:
        content (str): The actual text content of this chunk
        metadata (dict): Additional information (filename, document ID, chunk number, etc.)
        chunk_index (int): Position of this chunk in the document (0-based)
    
    Example:
        chunk = DocumentChunk(
            content="FastAPI is a modern web framework...",
            metadata={"filename": "fastapi_guide.pdf", "document_id": "123"},
            chunk_index=0
        )
    """
    content: str = Field(..., description="The text content of the chunk")
    # ... means this field is required (no default value)
    
    metadata: dict = Field(default_factory=dict, description="Metadata associated with the chunk")
    # default_factory=dict means empty dict {} if not provided
    
    chunk_index: int = Field(..., description="Index of the chunk within the document")


class DocumentUploadResponse(BaseModel):
    """
    Response schema returned after successfully uploading a document.
    
    When a document is uploaded, it's processed and indexed. This response
    confirms the upload was successful and provides information about the
    processing result.
    
    Attributes:
        document_id (str): Unique identifier for this document (UUID format)
        filename (str): Original name of the uploaded file
        num_chunks (int): How many chunks the document was split into
        status (str): Upload status ("success", "partial", "failed")
        message (str): Human-readable message about the result
        created_at (datetime): When the document was uploaded
    
    Example Response:
        {
            "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "filename": "fastapi_docs.pdf",
            "num_chunks": 42,
            "status": "success",
            "message": "Document processed and indexed successfully",
            "created_at": "2024-01-15T14:30:45"
        }
    """
    document_id: str = Field(..., description="Unique identifier for the uploaded document")
    filename: str = Field(..., description="Name of the uploaded file")
    num_chunks: int = Field(..., description="Number of chunks created from the document")
    status: str = Field(default="success", description="Status of the upload")
    message: str = Field(..., description="Additional message")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of upload")


# ============================================================================
# QUERY & RETRIEVAL SCHEMAS
# Used for querying the RAG system and returning results
# ============================================================================

class RetrievedDocument(BaseModel):
    """
    Schema for a document chunk retrieved from the vector database.
    
    When the RAG system retrieves relevant documents for a query, each result
    is returned with its content, relevance score, and metadata about its source.
    
    Attributes:
        content (str): The text snippet that matched the query
        score (float): Similarity score between 0 and 1 (higher = more relevant)
        document_id (str): Which document this chunk came from
        metadata (dict): Information about the source (filename, chunk index, etc.)
    
    Example:
        retrieved = RetrievedDocument(
            content="FastAPI provides automatic API documentation...",
            score=0.87,  # 87% similarity to query
            document_id="doc-123",
            metadata={"filename": "fastapi_guide.pdf", "chunk_index": 5}
        )
    """
    content: str = Field(..., description="The text content")
    score: float = Field(..., description="Similarity score (0-1, higher is better)")
    document_id: str = Field(..., description="Source document ID")
    metadata: dict = Field(default_factory=dict, description="Associated metadata")


class QueryRequest(BaseModel):
    """
    Request schema for RAG query endpoint.
    
    Users send this data to ask questions. The API validates the query before
    processing it through the RAG pipeline.
    
    Attributes:
        query (str): The question or search term (required, must not be empty)
        top_k (int): How many documents to retrieve (1-20, default 5)
        include_sources (bool): Whether to return source documents with the answer
        stream (bool): If True, stream response incrementally (for long answers)
    
    Validation Rules:
        - query: min_length=1 means empty strings are rejected
        - top_k: ge=1 (greater than or equal), le=20 (less than or equal)
          prevents requesting 0 results or millions of results
    
    Example Request JSON:
        {
            "query": "What is FastAPI?",
            "top_k": 3,
            "include_sources": true,
            "stream": false
        }
    """
    query: str = Field(..., min_length=1, description="The user query")
    # min_length=1 rejects empty strings; Pydantic validates automatically
    
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    # Optional means can be None; ge/le set minimum/maximum values
    # If not provided, defaults to 5
    
    include_sources: bool = Field(default=True, description="Whether to include source documents")
    stream: bool = Field(default=False, description="Whether to stream the response")


class QueryResponse(BaseModel):
    """
    Response schema for RAG query results.
    
    The API returns this structure after processing a query through the RAG
    pipeline. It includes the generated answer, source documents used, and
    metadata about the query processing.
    
    Attributes:
        query (str): Echo back the original query
        answer (str): Generated answer from the LLM based on retrieved documents
        sources (List[RetrievedDocument]): Document chunks used for the answer
        model (str): Which LLM model was used (e.g., "mixtral-8x7b-32768")
        processing_time (float): How long the query took (seconds)
        tokens_used (dict): Token usage if available (for cost tracking)
    
    Example Response JSON:
        {
            "query": "What is FastAPI?",
            "answer": "FastAPI is a modern, fast web framework for building APIs...",
            "sources": [
                {
                    "content": "FastAPI is a modern...",
                    "score": 0.92,
                    "document_id": "doc-1",
                    "metadata": {...}
                }
            ],
            "model": "mixtral-8x7b-32768",
            "processing_time": 2.35,
            "tokens_used": {"input": 150, "output": 200}
        }
    """
    query: str = Field(..., description="The original query")
    answer: str = Field(..., description="Generated answer from the LLM")
    sources: List[RetrievedDocument] = Field(default_factory=list, description="Retrieved source documents")
    model: str = Field(..., description="LLM model used")
    processing_time: float = Field(..., description="Time taken to process (seconds)")
    tokens_used: Optional[dict] = Field(default=None, description="Token usage if available")


# ============================================================================
# EMBEDDING SCHEMAS
# Used for internal embedding operations (rarely exposed to users)
# ============================================================================

class EmbeddingRequest(BaseModel):
    """
    Request schema for generating embeddings.
    
    This is used internally when documents need to be converted to vectors
    for storage in the vector database.
    
    Attributes:
        texts (List[str]): List of text snippets to convert to vectors
    
    Example:
        request = EmbeddingRequest(
            texts=["FastAPI is great", "I love Python"]
        )
    """
    texts: List[str] = Field(..., description="List of texts to embed")


class EmbeddingResponse(BaseModel):
    """
    Response schema for embedding results.
    
    Contains the generated vectors and metadata about the embedding operation.
    
    Attributes:
        embeddings (List[List[float]]): The vector representations (n_texts x vector_dimension)
        dimension (int): Size of each vector (e.g., 384 for all-MiniLM-L6-v2)
        model (str): Which embedding model was used
    
    Example:
        response = EmbeddingResponse(
            embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],  # Two vectors
            dimension=384,
            model="all-MiniLM-L6-v2"
        )
    """
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    dimension: int = Field(..., description="Dimension of embeddings")
    model: str = Field(..., description="Embedding model used")


# ============================================================================
# HEALTH & MONITORING SCHEMAS
# Used for system health checks and status monitoring
# ============================================================================

class HealthCheck(BaseModel):
    """
    Response schema for system health check endpoint.
    
    Clients can call GET /api/v1/health to check if the system is running
    and all services are available.
    
    Attributes:
        status (str): Overall health ("healthy", "degraded", "unhealthy")
        version (str): Application version number
        services (dict): Status of each major component
        timestamp (datetime): When the health check was performed
    
    Example Response:
        {
            "status": "healthy",
            "version": "1.0.0",
            "services": {
                "embedding_service": "ok",
                "llm_service": "ok",
                "vector_db_service": "ok"
            },
            "timestamp": "2024-01-15T14:30:45"
        }
    
    How to interpret:
        - status="healthy": All services OK
        - status="degraded": Some services working, some not
        - status="unhealthy": Major failure, many services down
    """
    status: str = Field(..., description="Overall status")
    version: str = Field(..., description="Application version")
    services: dict = Field(default_factory=dict, description="Status of each service")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of check")


# ============================================================================
# ERROR SCHEMAS
# Used for error responses
# ============================================================================

class ErrorDetail(BaseModel):
    """
    Schema for error responses.
    
    When something goes wrong, the API returns this structure with details
    about the error so clients know what happened and how to fix it.
    
    Attributes:
        error (str): Error type/category (e.g., "ValidationError", "NotFound")
        message (str): Human-readable error message
        details (dict): Additional error context (field names, validation rules, etc.)
        timestamp (datetime): When the error occurred
    
    Example Error Response:
        {
            "error": "ValidationError",
            "message": "Query cannot be empty",
            "details": {"field": "query", "constraint": "min_length"},
            "timestamp": "2024-01-15T14:30:45"
        }
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the error occurred")
