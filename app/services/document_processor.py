"""
Document processing service.
Handles PDF and Markdown parsing, chunking, and text extraction.
"""
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger("docmind")


class DocumentProcessor:
    """Service for processing and chunking documents."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_formats = {'.pdf', '.txt', '.md', '.markdown'}
    
    def process_document(self, file_path: str) -> Tuple[List[str], Dict]:
        """
        Process a document and return chunks with metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (chunks, metadata)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Processing document: {file_path.name}")
        
        if file_path.suffix.lower() == '.pdf':
            content = self._extract_pdf(str(file_path))
        elif file_path.suffix.lower() in {'.md', '.markdown'}:
            content = self._extract_markdown(str(file_path))
        else:
            content = self._extract_text(str(file_path))
        
        chunks = self._chunk_text(content)
        metadata = {
            'filename': file_path.name,
            'file_size': file_path.stat().st_size,
            'num_chunks': len(chunks)
        }
        
        logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
        return chunks, metadata
    
    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            import PyPDF2
        except ImportError:
            logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
            return ""
        
        text = []
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text.append(page.extract_text())
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
        
        return "\n".join(text)
    
    def _extract_markdown(self, file_path: str) -> str:
        """Extract text from Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading Markdown file: {e}")
            return ""
    
    def _extract_text(self, file_path: str) -> str:
        """Extract text from plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            return ""
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def process_batch(self, directory: str) -> Dict[str, Tuple[List[str], Dict]]:
        """
        Process all documents in a directory.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            Dictionary mapping filenames to (chunks, metadata)
        """
        results = {}
        dir_path = Path(directory)
        
        for file_path in dir_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    chunks, metadata = self.process_document(str(file_path))
                    results[file_path.name] = (chunks, metadata)
                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")
        
        return results
