"""
Unit tests for document processor service.
"""
import pytest
from app.services.document_processor import DocumentProcessor


def test_document_processor_initialization():
    """Test DocumentProcessor initialization."""
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    assert processor.chunk_size == 512
    assert processor.chunk_overlap == 50


def test_chunk_text(sample_text):
    """Test text chunking."""
    processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
    chunks = processor._chunk_text(sample_text)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_chunk_text_empty():
    """Test chunking empty text."""
    processor = DocumentProcessor()
    chunks = processor._chunk_text("")
    
    assert chunks == []


def test_supported_formats():
    """Test supported file formats."""
    processor = DocumentProcessor()
    assert '.pdf' in processor.supported_formats
    assert '.txt' in processor.supported_formats
    assert '.md' in processor.supported_formats
