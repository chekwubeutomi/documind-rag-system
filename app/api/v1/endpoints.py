"""
FastAPI v1 endpoints.
Handles document upload, querying, and health checks.
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

logger = logging.getLogger("docmind")
router = APIRouter(prefix="/v1", tags=["v1"])

# Initialize services
document_processor = None
embedding_service = None
llm_service = None
vector_db_service = None


def initialize_services():
    """Initialize all services."""
    global document_processor, embedding_service, llm_service, vector_db_service
    
    logger.info("Initializing services...")
    
    document_processor = DocumentProcessor(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    
    embedding_service = EmbeddingService(
        model_name=settings.EMBEDDING_MODEL,
        cache_folder=settings.EMBEDDING_MODEL_CACHE
    )
    
    llm_service = LLMService(
        provider=settings.LLM_PROVIDER,
        api_key=settings.GROQ_API_KEY or settings.OPENAI_API_KEY,
        model=settings.GROQ_MODEL or settings.OPENAI_MODEL
    )
    
    vector_db_service = VectorDBService(
        db_type=settings.VECTOR_DB_TYPE,
        url=settings.VECTOR_DB_URL
    )
    
    # Create or get collection
    if embedding_service.get_dimension():
        vector_db_service.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vector_size=embedding_service.get_dimension()
        )
    
    logger.info("Services initialized successfully")


# ============= Health Check Endpoints =============

@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Check application health and service status."""
    services_status = {
        "embedding_service": "ok" if embedding_service and embedding_service.model else "error",
        "llm_service": "ok" if llm_service and llm_service.client else "error",
        "vector_db_service": "ok" if vector_db_service and vector_db_service.client else "error",
    }
    
    status = "healthy" if all(v == "ok" for v in services_status.values()) else "degraded"
    
    return HealthCheck(
        status=status,
        version=settings.APP_VERSION,
        services=services_status
    )


# ============= Document Upload Endpoints =============

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload and process a document.
    
    Supports: PDF, Markdown, TXT
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file size
    file_content = await file.read()
    if len(file_content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    # Validate file type
    allowed_extensions = {'.pdf', '.txt', '.md', '.markdown'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="File type not supported")
    
    try:
        # Save file temporarily
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Process document
        chunks, metadata = document_processor.process_document(file_path)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from document")
        
        # Generate embeddings and store in vector DB
        document_id = str(uuid.uuid4())
        embeddings = embedding_service.embed_texts(chunks)
        
        # Add metadata
        chunk_metadata = [
            {
                "document_id": document_id,
                "filename": file.filename,
                "chunk_index": i,
                **metadata
            }
            for i in range(len(chunks))
        ]
        
        # Store in vector DB
        vector_db_service.add_vectors(
            collection_name=settings.QDRANT_COLLECTION,
            vectors=embeddings.tolist(),
            texts=chunks,
            metadata=chunk_metadata
        )
        
        logger.info(f"Document uploaded: {document_id} ({file.filename})")
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            num_chunks=len(chunks),
            message="Document processed and indexed successfully"
        )
    
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= Query Endpoints =============

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system.
    
    Retrieves relevant documents and generates an answer using the LLM.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    start_time = time.time()
    
    try:
        # Embed the query
        query_embedding = embedding_service.embed_text(request.query)
        
        if query_embedding.size == 0:
            raise HTTPException(status_code=500, detail="Could not embed query")
        
        # Retrieve similar documents
        search_results = vector_db_service.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_embedding.tolist(),
            top_k=request.top_k
        )
        
        # Filter by similarity threshold
        filtered_results = [
            (text, score, meta) for text, score, meta in search_results
            if score >= settings.SIMILARITY_THRESHOLD
        ]
        
        if not filtered_results:
            logger.warning(f"No relevant documents found for query: {request.query}")
        
        # Build context from retrieved documents
        context = "\n\n".join([f"Source: {meta.get('filename', 'Unknown')}\n{text}" for text, score, meta in filtered_results[:request.top_k]])
        
        # Generate answer using LLM
        system_prompt = """You are a helpful assistant that answers questions based on the provided documents. 
        Be accurate and cite the sources. If the information is not in the documents, say so."""
        
        prompt = f"""Based on the following documents, answer the question:

Documents:
{context}

Question: {request.query}

Answer:"""
        
        answer = llm_service.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1024,
            stream=request.stream
        )
        
        processing_time = time.time() - start_time
        
        # Build response
        sources = [
            RetrievedDocument(
                content=text,
                score=score,
                document_id=meta.get("document_id", "unknown"),
                metadata=meta
            )
            for text, score, meta in filtered_results if request.include_sources
        ]
        
        return QueryResponse(
            query=request.query,
            answer=answer if isinstance(answer, str) else "Stream response",
            sources=sources,
            model=settings.GROQ_MODEL or settings.OPENAI_MODEL,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
