"""
Script to batch ingest documents into the vector database.

This script is useful for:
- Initial knowledge base setup (large batch of documents)
- Periodic document ingestion (daily/weekly updates)
- CI/CD pipelines (automated document indexing)
- Development and testing (quickly populate vector DB)

Why a separate script?
- API upload endpoint processes one file at a time
- This script processes entire directories efficiently
- Can run as scheduled job or background worker
- Better for large-scale document ingestion

Workflow:
1. User provides directory path with documents
2. Script finds all PDF/TXT/MD files recursively
3. For each file:
   - Extract text and chunk
   - Generate embeddings
   - Store in vector database
4. Report final statistics (files processed, chunks created)

Execution:
$ python scripts/ingest_documents.py ./data/documents/
$ python scripts/ingest_documents.py ./data/documents/ --collection knowledge_base
"""
import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path (so we can import app modules)
# When running scripts, we're in scripts/ directory but need to import from app/
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logging import logger
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_db_service import VectorDBService


def ingest_documents(input_dir: str, collection_name: str = None):
    """
    Ingest all documents from a directory into the vector database.
    
    This is the main function that orchestrates batch document ingestion.
    It processes all documents in a directory (recursively) and stores
    them in the vector database for RAG retrieval.
    
    Ingestion Pipeline:
    1. Find all documents (*.pdf, *.txt, *.md)
    2. For each document:
       a. Extract text using DocumentProcessor
       b. Generate embeddings using EmbeddingService
       c. Store in vector database using VectorDBService
    3. Report statistics
    
    Args:
        input_dir (str): Directory containing documents
            Can include subdirectories (searches recursively)
            Example: "./data/documents" or "/mnt/documents"
            
        collection_name (str): Which collection to store in
            Default: uses settings.QDRANT_COLLECTION
            Example: "documents" or "knowledge_base"
    
    Log Output:
        - Starting message with directory
        - Processing status for each file (filename, chunks)
        - Summary with total documents and chunks
        - Errors logged with filenames for debugging
    
    Example Output:
        INFO Starting document ingestion from: ./data/documents
        INFO Creating collection: documents (dimension: 384)
        INFO Processing: guide.pdf
        INFO Generating embeddings for 42 chunks...
        INFO ✓ Successfully indexed guide.pdf
        ...
        INFO Ingestion complete!
        INFO Documents processed: 5
        INFO Total chunks created: 234
    """
    # Use provided collection name or fall back to settings default
    collection_name = collection_name or settings.QDRANT_COLLECTION
    
    logger.info(f"Starting document ingestion from: {input_dir}")
    
    # ======================================================================
    # STEP 1: Initialize Services
    # ======================================================================
    # Create instances of the three core services we need
    logger.info("Initializing services...")
    
    # Document processor: extracts and chunks text
    document_processor = DocumentProcessor(
        chunk_size=settings.CHUNK_SIZE,  # Default 512
        chunk_overlap=settings.CHUNK_OVERLAP  # Default 50
    )
    
    # Embedding service: converts text to vectors
    embedding_service = EmbeddingService(
        model_name=settings.EMBEDDING_MODEL,  # Default all-MiniLM-L6-v2
        cache_folder=settings.EMBEDDING_MODEL_CACHE
    )
    
    # Vector DB service: stores vectors and metadata
    vector_db_service = VectorDBService(
        db_type=settings.VECTOR_DB_TYPE,  # Default qdrant
        url=settings.VECTOR_DB_URL  # Default http://localhost:6333
    )
    
    # ======================================================================
    # STEP 2: Create Collection
    # ======================================================================
    # Create the collection that will store all vectors
    vector_dim = embedding_service.get_dimension()  # Get model's output dimension
    logger.info(f"Creating collection: {collection_name} (dimension: {vector_dim})")
    vector_db_service.create_collection(collection_name, vector_dim)
    
    # ======================================================================
    # STEP 3: Process All Documents
    # ======================================================================
    # Find all documents and process them
    input_path = Path(input_dir)
    
    # Check if directory exists
    if not input_path.exists():
        logger.error(f"Directory not found: {input_dir}")
        return
    
    # Track statistics
    documents_processed = 0
    chunks_created = 0
    
    # rglob("*"): recursive glob finds all files in all subdirectories
    for file_path in input_path.rglob("*"):
        # Skip if it's a directory (we only want files)
        if not file_path.is_file():
            continue
        
        # Skip files with unsupported extensions
        if file_path.suffix.lower() not in {'.pdf', '.txt', '.md', '.markdown'}:
            continue
        
        try:
            logger.info(f"Processing: {file_path.name}")
            
            # ============================================================
            # Process Document: Extract and chunk text
            # ============================================================
            chunks, metadata = document_processor.process_document(str(file_path))
            
            # Skip if no text was extracted (empty file?)
            if not chunks:
                logger.warning(f"No chunks extracted from {file_path.name}")
                continue
            
            # ============================================================
            # Generate Embeddings
            # ============================================================
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = embedding_service.embed_texts(chunks)
            
            # ============================================================
            # Prepare Metadata
            # ============================================================
            # Create metadata for each chunk
            # Metadata helps track where chunks came from
            chunk_metadata = [
                {
                    "filename": file_path.name,  # Original filename
                    "file_path": str(file_path),  # Full path
                    "chunk_index": i,  # Which chunk number (0, 1, 2...)
                    **metadata  # Include file-level metadata (size, etc)
                }
                for i in range(len(chunks))
            ]
            
            # ============================================================
            # Add to Vector Database
            # ============================================================
            success = vector_db_service.add_vectors(
                collection_name=collection_name,
                vectors=embeddings.tolist(),  # Convert numpy to list
                texts=chunks,  # Original text chunks
                metadata=chunk_metadata  # Metadata for tracking
            )
            
            # Track success
            if success:
                documents_processed += 1
                chunks_created += len(chunks)
                logger.info(f"✓ Successfully indexed {file_path.name}")
            else:
                logger.error(f"✗ Failed to index {file_path.name}")
        
        except Exception as e:
            # Log error but continue with next file
            logger.error(f"Error processing {file_path.name}: {e}")
    
    # ======================================================================
    # STEP 4: Report Summary
    # ======================================================================
    logger.info(f"\nIngestion complete!")
    logger.info(f"Documents processed: {documents_processed}")
    logger.info(f"Total chunks created: {chunks_created}")


def main():
    """
    Main entry point for command-line usage.
    
    Parses command-line arguments and calls ingest_documents().
    
    Usage:
        python scripts/ingest_documents.py ./data/documents/
        python scripts/ingest_documents.py /path/to/docs --collection my_collection
    
    Arguments:
        input_dir: Directory containing documents (required)
        --collection: Collection name (optional, defaults to settings)
    """
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Ingest documents into the vector database"
    )
    
    # Positional argument: directory
    parser.add_argument(
        "input_dir",
        help="Directory containing documents to ingest"
    )
    
    # Optional argument: collection name
    parser.add_argument(
        "--collection",
        default=settings.QDRANT_COLLECTION,
        help="Name of the collection"
    )
    
    # Parse arguments from command line
    args = parser.parse_args()
    
    try:
        # Call the main ingestion function
        ingest_documents(args.input_dir, args.collection)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)  # Exit with error code


if __name__ == "__main__":
    main()
