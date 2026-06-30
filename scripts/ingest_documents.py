"""
Script to ingest documents into the vector database.
"""
import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logging import logger
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_db_service import VectorDBService


def ingest_documents(input_dir: str, collection_name: str = None):
    """
    Ingest all documents from a directory into the vector database.
    
    Args:
        input_dir: Directory containing documents to ingest
        collection_name: Name of the collection (uses settings if not provided)
    """
    collection_name = collection_name or settings.QDRANT_COLLECTION
    
    logger.info(f"Starting document ingestion from: {input_dir}")
    
    # Initialize services
    logger.info("Initializing services...")
    document_processor = DocumentProcessor(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    
    embedding_service = EmbeddingService(
        model_name=settings.EMBEDDING_MODEL,
        cache_folder=settings.EMBEDDING_MODEL_CACHE
    )
    
    vector_db_service = VectorDBService(
        db_type=settings.VECTOR_DB_TYPE,
        url=settings.VECTOR_DB_URL
    )
    
    # Create collection
    vector_dim = embedding_service.get_dimension()
    logger.info(f"Creating collection: {collection_name} (dimension: {vector_dim})")
    vector_db_service.create_collection(collection_name, vector_dim)
    
    # Process all documents in directory
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Directory not found: {input_dir}")
        return
    
    documents_processed = 0
    chunks_created = 0
    
    for file_path in input_path.rglob("*"):
        if not file_path.is_file():
            continue
        
        if file_path.suffix.lower() not in {'.pdf', '.txt', '.md', '.markdown'}:
            continue
        
        try:
            logger.info(f"Processing: {file_path.name}")
            
            # Process document
            chunks, metadata = document_processor.process_document(str(file_path))
            
            if not chunks:
                logger.warning(f"No chunks extracted from {file_path.name}")
                continue
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = embedding_service.embed_texts(chunks)
            
            # Prepare metadata
            chunk_metadata = [
                {
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "chunk_index": i,
                    **metadata
                }
                for i in range(len(chunks))
            ]
            
            # Add to vector DB
            success = vector_db_service.add_vectors(
                collection_name=collection_name,
                vectors=embeddings.tolist(),
                texts=chunks,
                metadata=chunk_metadata
            )
            
            if success:
                documents_processed += 1
                chunks_created += len(chunks)
                logger.info(f"✓ Successfully indexed {file_path.name}")
            else:
                logger.error(f"✗ Failed to index {file_path.name}")
        
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
    
    logger.info(f"\nIngestion complete!")
    logger.info(f"Documents processed: {documents_processed}")
    logger.info(f"Total chunks created: {chunks_created}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ingest documents into the vector database")
    parser.add_argument(
        "input_dir",
        help="Directory containing documents to ingest"
    )
    parser.add_argument(
        "--collection",
        default=settings.QDRANT_COLLECTION,
        help="Name of the collection"
    )
    
    args = parser.parse_args()
    
    try:
        ingest_documents(args.input_dir, args.collection)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
