"""
FastAPI v1 API endpoints module.

This module defines all REST API endpoints for the DocMind RAG application.
It handles three main operations:
1. Health checks - Verify system is running and services are healthy
2. Document uploads - Ingest documents into the RAG system
3. Queries - Retrieve relevant documents and generate AI answers

API Design Pattern: Request → Service Layer → Response
- Endpoints receive HTTP requests with validation (Pydantic schemas)
- Delegate work to service layer (document processor, embeddings, LLM, vector DB)
- Return JSON responses with appropriate HTTP status codes

FastAPI Features Used:
- APIRouter: Group related endpoints with shared prefix (/v1)
- UploadFile: Handle file uploads with automatic cleanup
- BackgroundTasks: Run tasks after returning response
- StreamingResponse: Stream responses token-by-token
- HTTPException: Return standard HTTP errors
- Pydantic models: Auto-validate requests and responses (generates OpenAPI docs)

Workflow Examples:
Upload Flow:
    User uploads PDF → Validate → Extract text → Split chunks → Generate embeddings → Store in vector DB

Query Flow:
    User asks question → Embed query → Search vectors → Get top-K documents → Build context → Call LLM → Return answer

Error Handling:
- File validation: size, format, readability
- Service initialization: check services ready before use
- Search failures: return empty context or error message
- LLM failures: timeout after 30 seconds, return gracefully
"""
import logging
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.schemas.request_response import (
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
    HealthCheck,
    ErrorDetail,
    RetrievedDocument
)
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.vector_db_service import VectorDBService
import time
import uuid
import os

logger = logging.getLogger("docmind")  # Get logger for this module
router = APIRouter(prefix="/v1", tags=["v1"])  # Router for all /v1/* endpoints

# ======================================================================
# Global Service Instances
# ======================================================================
# These are initialized by initialize_services() when the app starts
# They're kept as globals so all endpoints can access them
document_processor = None  # Processes PDFs/markdown and chunks text
embedding_service = None  # Converts text to embeddings
llm_service = None  # Generates answers using LLM
vector_db_service = None  # Stores and retrieves vectors


def initialize_services():
    """
    Initialize all services during application startup.
    
    This function is called by the FastAPI lifespan event on startup.
    It creates instances of all services and stores them in global variables.
    If any service fails to initialize, a warning is logged but the app continues
    (graceful degradation - that feature will fail but others will work).
    
    Services Initialized:
    1. DocumentProcessor: Text extraction and chunking
    2. EmbeddingService: Text to vector conversion
    3. LLMService: Answer generation
    4. VectorDBService: Vector storage and retrieval
    
    Also creates the Qdrant collection if it doesn't exist.
    
    Call Flow:
    app.py defines lifespan context → on startup, calls this → global services ready
    Then all endpoints can call these global service instances
    """
    global document_processor, embedding_service, llm_service, vector_db_service
    
    logger.info("Initializing services...")
    
    # ======================================================================
    # Initialize DocumentProcessor
    # ======================================================================
    # This service extracts text from PDFs/Markdown and chunks it
    document_processor = DocumentProcessor(
        chunk_size=settings.CHUNK_SIZE,  # Default: 512 characters per chunk
        chunk_overlap=settings.CHUNK_OVERLAP  # Default: 50 character overlap
    )
    
    # ======================================================================
    # Initialize EmbeddingService
    # ======================================================================
    # This service converts text to vectors using sentence-transformers
    embedding_service = EmbeddingService(
        model_name=settings.EMBEDDING_MODEL,  # Default: all-MiniLM-L6-v2
        cache_folder=settings.EMBEDDING_MODEL_CACHE  # Where to cache model files
    )
    
    # ======================================================================
    # Initialize LLMService
    # ======================================================================
    # This service generates answers using LLM providers (Groq, Ollama, OpenAI)
    llm_service = LLMService(
        provider=settings.LLM_PROVIDER,  # Default: groq (fast)
        api_key=settings.GROQ_API_KEY or settings.OPENAI_API_KEY,  # API auth
        model=settings.GROQ_MODEL or settings.OPENAI_MODEL  # Default model to use
    )
    
    # ======================================================================
    # Initialize VectorDBService
    # ======================================================================
    # This service stores and retrieves embeddings from vector database
    vector_db_service = VectorDBService(
        db_type=settings.VECTOR_DB_TYPE,  # Default: qdrant
        url=settings.VECTOR_DB_URL  # Default: http://localhost:6333
    )
    
    # ======================================================================
    # Create Vector Database Collection
    # ======================================================================
    # Collections are like tables in traditional databases
    # One collection for all documents in this RAG system
    # Vector dimension must match the embedding model's output
    if embedding_service.get_dimension():
        dimension = embedding_service.get_dimension()  # e.g., 384 for default model
        vector_db_service.create_collection(
            collection_name=settings.QDRANT_COLLECTION,  # e.g., "documents"
            vector_size=dimension  # 384 dimensions for all-MiniLM
        )
    
    logger.info("Services initialized successfully")


# ======================================================================
# HEALTH CHECK ENDPOINTS
# ======================================================================
# These endpoints let external services check if DocMind is running
# Used by docker-compose health checks, Kubernetes probes, monitoring systems

@router.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Health check endpoint for monitoring system status.
    
    This endpoint returns the current status of the application and its services.
    Used by:
    - Docker health checks: keeps container alive if healthy
    - Kubernetes liveness probes: restarts pod if not healthy
    - Monitoring systems: alerts on failures
    - Load balancers: routes traffic away from unhealthy instances
    
    HTTP Status:
    - 200 OK: System is healthy or degraded (at least one service working)
    - 503 Service Unavailable: Could return this for all services down
    
    Returns:
        HealthCheck object with:
        - status: "healthy" (all services ok) or "degraded" (some services down)
        - version: Application version (e.g., "0.1.0")
        - services: Dict showing status of each service ("ok" or "error")
        - timestamp: When this check was performed
    
    Example Response:
        {
            "status": "healthy",
            "version": "0.1.0",
            "services": {
                "embedding_service": "ok",
                "llm_service": "ok",
                "vector_db_service": "ok"
            },
            "timestamp": "2024-01-15T10:30:45.123456"
        }
    """
    # Check status of each service
    services_status = {
        "embedding_service": "ok" if embedding_service and embedding_service.model else "error",
        "llm_service": "ok" if llm_service and llm_service.client else "error",
        "vector_db_service": "ok" if vector_db_service and vector_db_service.client else "error",
    }
    
    # Overall status: "healthy" if all ok, "degraded" if any failing
    status = "healthy" if all(v == "ok" for v in services_status.values()) else "degraded"
    
    return HealthCheck(
        status=status,
        version=settings.APP_VERSION,
        services=services_status
    )


# ======================================================================
# DOCUMENT UPLOAD ENDPOINTS
# ======================================================================
# These endpoints handle document ingestion into the RAG system

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload a document for RAG ingestion.
    
    This endpoint handles the complete document ingestion pipeline:
    1. Validate file format and size
    2. Save file temporarily
    3. Extract text (PDF, Markdown, or TXT)
    4. Split into chunks with overlap
    5. Generate embeddings for each chunk
    6. Store in vector database
    
    Document Upload Pipeline:
    
    User selects file
         ↓
    Upload → Validate (size, format)
         ↓
    Save temporarily
         ↓
    DocumentProcessor.process_document() → Extract text + chunk
         ↓
    EmbeddingService.embed_texts() → Generate embeddings (vectors)
         ↓
    VectorDBService.add_vectors() → Store in database
         ↓
    Success response with document ID + metrics
    
    Supported Formats:
    - .pdf: Binary PDF files (requires PyPDF2)
    - .txt: Plain text files (UTF-8)
    - .md, .markdown: Markdown files (UTF-8 with markdown syntax)
    
    Validation Checks:
    - File must be provided (400 Bad Request if missing)
    - File size must be under MAX_UPLOAD_SIZE (default 50MB) (413 Payload Too Large)
    - File extension must be in allowed list (400 Bad Request)
    
    HTTP Status Codes:
    - 200 OK: Document successfully uploaded and indexed
    - 400 Bad Request: Invalid file type, empty file, or no file provided
    - 413 Payload Too Large: File exceeds MAX_UPLOAD_SIZE limit
    - 500 Internal Server Error: Server error during processing
    
    Args:
        file (UploadFile): The document file to upload
            The file is automatically cleaned up by FastAPI after response
        
        background_tasks (BackgroundTasks): Optional background tasks
            Could be used for async cleanup or post-processing
    
    Returns:
        DocumentUploadResponse with:
        - document_id: UUID for tracking this document
        - filename: Original filename
        - num_chunks: How many text chunks were created
        - status: "completed" or "processing"
        - message: Human-readable status message
        - created_at: When document was uploaded
    
    Example Request:
        POST /v1/documents/upload
        Content-Type: multipart/form-data
        
        file: <PDF file>
    
    Example Response:
        {
            "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "filename": "machine_learning_guide.pdf",
            "num_chunks": 42,
            "status": "completed",
            "message": "Document processed and indexed successfully",
            "created_at": "2024-01-15T10:30:45.123456"
        }
    """
    # ======================================================================
    # STEP 1: Validate file provided
    # ======================================================================
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # ======================================================================
    # STEP 2: Validate file size
    # ======================================================================
    file_content = await file.read()  # Read entire file into memory
    if len(file_content) > settings.MAX_UPLOAD_SIZE:
        max_size_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {max_size_mb}MB"
        )
    
    # ======================================================================
    # STEP 3: Validate file type
    # ======================================================================
    allowed_extensions = {'.pdf', '.txt', '.md', '.markdown'}
    file_ext = os.path.splitext(file.filename)[1].lower()  # Get extension
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {allowed_extensions}"
        )
    
    try:
        # ======================================================================
        # STEP 4: Save file temporarily
        # ======================================================================
        # Create upload directory if it doesn't exist
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        # Save file to disk (document_processor will read from here)
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        logger.info(f"File saved: {file_path}")
        
        # ======================================================================
        # STEP 5: Extract text and chunk
        # ======================================================================
        # DocumentProcessor handles format detection and text extraction
        chunks, metadata = document_processor.process_document(file_path)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from document"
            )
        
        logger.info(f"Extracted {len(chunks)} chunks from {file.filename}")
        
        # ======================================================================
        # STEP 6: Generate embeddings
        # ======================================================================
        # Convert all chunks to vectors (384 dimensions each)
        # Returns numpy array of shape (num_chunks, 384)
        embeddings = embedding_service.embed_texts(chunks)
        
        # ======================================================================
        # STEP 7: Add metadata to each chunk
        # ======================================================================
        # Metadata helps track where each vector came from
        document_id = str(uuid.uuid4())  # Generate unique ID for this document
        chunk_metadata = [
            {
                "document_id": document_id,  # Which document this chunk came from
                "filename": file.filename,  # Original filename
                "chunk_index": i,  # Which chunk number this is (0, 1, 2, ...)
                **metadata  # Include file metadata (size, etc)
            }
            for i in range(len(chunks))
        ]
        
        # ======================================================================
        # STEP 8: Store in vector database
        # ======================================================================
        # This makes the vectors searchable
        success = vector_db_service.add_vectors(
            collection_name=settings.QDRANT_COLLECTION,  # Default: "documents"
            vectors=embeddings.tolist(),  # Convert numpy array to list
            texts=chunks,  # Original text chunks
            metadata=chunk_metadata  # Metadata for each chunk
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to store vectors in database"
            )
        
        logger.info(f"Document {document_id} indexed with {len(chunks)} chunks")
        
        # ======================================================================
        # STEP 9: Return success response
        # ======================================================================
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            num_chunks=len(chunks),
            message="Document processed and indexed successfully"
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is (they already have status codes)
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================================
# QUERY ENDPOINTS
# ======================================================================
# These endpoints handle RAG queries (retrieve + generate answers)

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system and get AI-generated answers.
    
    This is the main endpoint for end-users. It implements the complete RAG pipeline:
    Retrieve + Generate
    
    Complete RAG Pipeline:
    
    User Question
         ↓
    EmbeddingService.embed_text() → Convert question to vector (384 floats)
         ↓
    VectorDBService.search() → Find K most similar documents (similarity search)
         ↓
    Retrieve top-K relevant documents (text + metadata + similarity scores)
         ↓
    Filter by SIMILARITY_THRESHOLD (only keep score > 0.5)
         ↓
    Build context: combine all relevant documents into one string
         ↓
    Build prompt: "Based on documents {context}, answer: {question}"
         ↓
    LLMService.generate() → Send to LLM (Groq/OpenAI/Ollama)
         ↓
    LLM generates answer using the retrieved context
         ↓
    Return answer + source documents + processing time
    
    Why This Works:
    1. Embedding converts question to vector in semantic space
    2. Vector DB finds semantically similar documents (not just keyword match)
    3. Documents provide factual context the LLM can use
    4. LLM is prompted to cite sources and stay accurate
    5. Result: Accurate, grounded, traceable answers (not hallucinations!)
    
    HTTP Status Codes:
    - 200 OK: Query processed successfully
    - 400 Bad Request: Empty query string
    - 500 Internal Server Error: Processing error
    
    Args:
        request (QueryRequest) containing:
        - query (str): The question to answer (required)
        - top_k (int): How many documents to retrieve (default 5, range 1-20)
        - include_sources (bool): Include source documents in response (default false)
        - stream (bool): Stream response token-by-token (default false)
    
    Returns:
        QueryResponse with:
        - query: Echo back the original query
        - answer: AI-generated answer
        - sources: List of retrieved documents (if include_sources=true)
        - model: Which LLM model was used
        - processing_time: How long it took (seconds)
        - tokens_used: Approximate token count
    
    Example Request:
        POST /v1/query
        Content-Type: application/json
        
        {
            "query": "What is machine learning?",
            "top_k": 5,
            "include_sources": true,
            "stream": false
        }
    
    Example Response:
        {
            "query": "What is machine learning?",
            "answer": "Machine learning is a subset of AI that enables computers to learn from data...",
            "sources": [
                {
                    "content": "ML is the study of algorithms that improve through experience",
                    "score": 0.92,
                    "document_id": "doc-123",
                    "metadata": {"filename": "ml_guide.pdf", "page": 5}
                }
            ],
            "model": "mixtral-8x7b-32768",
            "processing_time": 2.34,
            "tokens_used": 450
        }
    """
    # ======================================================================
    # STEP 1: Validate query
    # ======================================================================
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info(f"Query received: {request.query[:100]}...")
    
    # Record start time for processing_time metric
    start_time = time.time()
    
    try:
        # ======================================================================
        # STEP 2: Embed the query
        # ======================================================================
        # Convert question to 384-dimensional vector
        # This vector represents the semantic meaning of the question
        query_embedding = embedding_service.embed_text(request.query)
        
        if query_embedding.size == 0:
            raise HTTPException(status_code=500, detail="Could not embed query")
        
        logger.info(f"Query embedded (dimension: {query_embedding.shape})")
        
        # ======================================================================
        # STEP 3: Search for similar documents
        # ======================================================================
        # Find top-K vectors most similar to query vector
        # Qdrant HNSW search: fast (milliseconds even for millions of vectors!)
        search_results = vector_db_service.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_embedding.tolist(),  # Convert numpy array to list
            top_k=request.top_k  # User can specify how many results (1-20)
        )
        
        logger.info(f"Found {len(search_results)} documents")
        
        # ======================================================================
        # STEP 4: Filter by similarity threshold
        # ======================================================================
        # Only keep documents that are actually relevant (similarity score > threshold)
        # Prevents LLM from getting confused by barely-relevant documents
        # threshold = 0.5 means: "more similar than different"
        filtered_results = [
            (text, score, meta) for text, score, meta in search_results
            if score >= settings.SIMILARITY_THRESHOLD
        ]
        
        if not filtered_results:
            logger.warning(f"No relevant documents found for query: {request.query}")
            # Don't fail - just return answer with empty context
        
        # ======================================================================
        # STEP 5: Build context string
        # ======================================================================
        # Concatenate all retrieved documents with source info
        # Format: "Source: filename\nText..."
        # This becomes the RAG context that the LLM will use
        context = "\n\n".join([
            f"Source: {meta.get('filename', 'Unknown')}\n{text}" 
            for text, score, meta in filtered_results[:request.top_k]
        ])
        
        if not context:
            context = "No relevant documents found in the database."
        
        # ======================================================================
        # STEP 6: Build prompt for LLM
        # ======================================================================
        # This is the prompt engineering that makes RAG work!
        # We tell the LLM:
        # 1. "Here's the information"
        # 2. "Answer based ONLY on this information"
        # 3. "Cite sources"
        system_prompt = """You are a helpful assistant that answers questions based on the provided documents. 
        Be accurate and cite the sources. If the information is not in the documents, say so."""
        
        # Build the full prompt with context
        prompt = f"""Based on the following documents, answer the question:

Documents:
{context}

Question: {request.query}

Answer:"""
        
        logger.info(f"Calling LLM with context ({len(context)} chars)")
        
        # ======================================================================
        # STEP 7: Call LLM to generate answer
        # ======================================================================
        # Send prompt to LLM (Groq, OpenAI, or Ollama)
        # LLM generates answer based on context
        # If stream=true, returns a stream to send token-by-token to user
        answer = llm_service.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7,  # Balanced: creative but accurate
            max_tokens=1024,  # Allow up to 1KB response
            stream=request.stream  # Stream tokens if requested
        )
        
        # ======================================================================
        # STEP 8: Calculate metrics
        # ======================================================================
        processing_time = time.time() - start_time
        
        logger.info(f"Query processed in {processing_time:.2f}s")
        
        # ======================================================================
        # STEP 9: Build response
        # ======================================================================
        # Include source documents if requested
        sources = [
            RetrievedDocument(
                content=text,
                score=score,
                document_id=meta.get("document_id", "unknown"),
                metadata=meta
            )
            for text, score, meta in filtered_results if request.include_sources
        ]
        
        # Return complete response
        return QueryResponse(
            query=request.query,
            answer=answer if isinstance(answer, str) else "Stream response",
            sources=sources,
            model=settings.GROQ_MODEL or settings.OPENAI_MODEL,
            processing_time=processing_time
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
