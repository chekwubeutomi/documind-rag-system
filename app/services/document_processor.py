"""
Document processing service module.

This module is responsible for extracting text from various document formats
(PDF, Markdown, plain text) and splitting them into manageable chunks for
embedding and indexing in the vector database.

Why Chunking is Important:
- Embedding models work best with chunks of moderate size (512-1024 tokens)
- Chunks allow granular retrieval: when searching, users get specific relevant sections
- Overlapping chunks prevent important information from being lost at chunk boundaries
- Smaller chunks = higher retrieval precision; larger chunks = more context

Supported Formats:
- .pdf: Portable Document Format (uses PyPDF2 for extraction)
- .txt: Plain text files (simple UTF-8 reading)
- .md, .markdown: Markdown files (also simple UTF-8 reading)

Architecture:
1. DocumentProcessor.process_document() - Main entry point
2. Format-specific extractors (_extract_pdf, _extract_markdown, _extract_text)
3. _chunk_text() - Splits extracted text into overlapping chunks
4. process_batch() - Processes entire directories of documents
"""
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger("docmind")  # Get logger for this module


class DocumentProcessor:
    """
    Service class for processing documents and extracting text chunks.
    
    This class handles the pipeline for converting raw documents into text chunks
    suitable for embedding and vector storage. It supports multiple document formats
    and provides both single-document and batch processing capabilities.
    
    Workflow:
    1. Validate file format and existence
    2. Extract text using appropriate method (PDF/MD/TXT)
    3. Split text into overlapping chunks
    4. Return chunks with metadata (filename, file size, chunk count)
    
    Example Usage:
        processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
        chunks, metadata = processor.process_document("document.pdf")
        print(f"Created {metadata['num_chunks']} chunks")
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the DocumentProcessor with chunking parameters.
        
        Args:
            chunk_size (int): Number of characters per chunk. Default 512.
                - 512 chars ≈ 100-150 words (good for RAG systems)
                - Larger = more context per chunk, fewer chunks total
                - Smaller = more granular retrieval, more chunks total
            
            chunk_overlap (int): Number of characters overlapping between chunks. Default 50.
                - Prevents important info from being split across chunk boundaries
                - Example: chunk 1 is chars 0-512, chunk 2 is chars 462-974
                - Higher overlap = better continuity, more chunks total
        """
        self.chunk_size = chunk_size  # Store for later use
        self.chunk_overlap = chunk_overlap  # Store for later use
        # Define which file extensions are supported
        self.supported_formats = {'.pdf', '.txt', '.md', '.markdown'}
    
    def process_document(self, file_path: str) -> Tuple[List[str], Dict]:
        """
        Main method: process a document and extract chunks with metadata.
        
        This is the primary entry point. It handles format detection, text extraction,
        chunking, and returns everything the RAG system needs.
        
        Args:
            file_path (str): Path to the document file (can be relative or absolute)
            
        Returns:
            Tuple[List[str], Dict]: 
                - List[str]: Text chunks ready for embedding
                - Dict: Metadata including filename, file size, chunk count
                
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
            
        Example:
            >>> processor = DocumentProcessor()
            >>> chunks, metadata = processor.process_document("guide.pdf")
            >>> print(f"Chunks: {len(chunks)}")
            Chunks: 42
            >>> print(metadata['filename'])
            guide.pdf
        """
        # Convert string path to Path object (more convenient for operations)
        file_path = Path(file_path)
        
        # ======================================================================
        # STEP 1: Validate file exists
        # ======================================================================
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # ======================================================================
        # STEP 2: Validate file format is supported
        # ======================================================================
        if file_path.suffix.lower() not in self.supported_formats:
            error_msg = f"Unsupported file format: {file_path.suffix}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Processing document: {file_path.name}")
        
        # ======================================================================
        # STEP 3: Extract text using format-specific method
        # ======================================================================
        if file_path.suffix.lower() == '.pdf':
            # PDF files require special handling with PyPDF2
            content = self._extract_pdf(str(file_path))
        elif file_path.suffix.lower() in {'.md', '.markdown'}:
            # Markdown files are just text with special formatting
            content = self._extract_markdown(str(file_path))
        else:
            # Plain text files (fallback for .txt and unknown formats)
            content = self._extract_text(str(file_path))
        
        # ======================================================================
        # STEP 4: Split extracted text into chunks
        # ======================================================================
        chunks = self._chunk_text(content)
        
        # ======================================================================
        # STEP 5: Create metadata dictionary
        # ======================================================================
        metadata = {
            'filename': file_path.name,  # Just the filename (not full path)
            'file_size': file_path.stat().st_size,  # File size in bytes
            'num_chunks': len(chunks)  # How many chunks were created
        }
        
        logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
        return chunks, metadata
    
    def _extract_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Uses PyPDF2 library to read PDF files page by page and extract text.
        
        Args:
            file_path (str): Path to PDF file
            
        Returns:
            str: Extracted text from all pages joined with newlines
            
        Error Handling:
            - If PyPDF2 isn't installed, returns empty string with warning
            - If PDF extraction fails, logs error and returns empty string
            - Graceful degradation: doesn't crash, just skips the file
        """
        # Check if PyPDF2 library is available
        try:
            import PyPDF2
        except ImportError:
            error_msg = "PyPDF2 not installed. Install with: pip install PyPDF2"
            logger.warning(error_msg)
            return ""  # Return empty string, don't crash
        
        text = []  # List to collect text from each page
        try:
            # Open PDF file in binary mode (required for PDF reading)
            with open(file_path, 'rb') as file:
                # Create a PDF reader object
                reader = PyPDF2.PdfReader(file)
                
                # Iterate through each page in the PDF
                for page in reader.pages:
                    # Extract text from this page
                    extracted = page.extract_text()
                    if extracted:  # Only add non-empty pages
                        text.append(extracted)
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            return ""  # Return empty string on error
        
        # Join all pages with newlines to preserve page boundaries
        return "\n".join(text)
    
    def _extract_markdown(self, file_path: str) -> str:
        """
        Extract text from a Markdown file.
        
        Markdown files are just plain text with special formatting syntax.
        We extract all content including the markdown syntax (it's useful for structure).
        
        Args:
            file_path (str): Path to Markdown file (.md or .markdown)
            
        Returns:
            str: Full text content of the file, or empty string on error
        """
        try:
            # Open file in text mode with UTF-8 encoding
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()  # Read entire file into memory
        except Exception as e:
            logger.error(f"Error reading Markdown file: {e}")
            return ""  # Return empty string on error
    
    def _extract_text(self, file_path: str) -> str:
        """
        Extract text from a plain text file.
        
        This is the simplest case - just read the file as-is.
        
        Args:
            file_path (str): Path to text file (.txt)
            
        Returns:
            str: Full file content, or empty string on error
        """
        try:
            # Open file in text mode with UTF-8 encoding
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()  # Read entire file
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            return ""  # Return empty string on error
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        This is a critical function. It breaks long text into manageable pieces
        that can be embedded and retrieved. Overlapping ensures info at boundaries
        isn't lost.
        
        Chunking Strategy - Sliding Window:
        If chunk_size=512 and overlap=50:
        - Chunk 0: chars 0-512
        - Chunk 1: chars 462-974 (starts 50 chars before previous chunk ended)
        - Chunk 2: chars 924-1436
        - etc.
        
        Visual example:
        [====== CHUNK 0 ======]
                       [====== CHUNK 1 ======]
                                      [====== CHUNK 2 ======]
                                             ^^^^^^
                                          overlap area
        
        Args:
            text (str): The full text to chunk (e.g., extracted document text)
            
        Returns:
            List[str]: List of text chunks, with empty/whitespace-only chunks removed
            
        Example:
            >>> processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
            >>> text = "This is a long document that needs to be split."
            >>> chunks = processor._chunk_text(text)
            >>> len(chunks)
            2
        """
        # Handle edge case: empty text
        if not text:
            return []
        
        chunks = []  # Will store our text chunks
        start = 0  # Current starting position
        
        # ======================================================================
        # Sliding Window Loop - Create overlapping chunks
        # ======================================================================
        while start < len(text):
            # Calculate end position of this chunk
            end = start + self.chunk_size
            
            # Extract the chunk (note: can exceed text length, Python handles it)
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position for next chunk
            # Moving by (chunk_size - overlap) creates overlap
            # Example: if chunk_size=100 and overlap=10, move by 90
            # This means last 10 chars of previous chunk are in next chunk
            start = end - self.chunk_overlap
        
        # ======================================================================
        # Post-processing: Remove empty/whitespace-only chunks
        # ======================================================================
        # Some chunks might be mostly whitespace (at file end)
        # Strip() removes leading/trailing whitespace; bool() returns False for empty strings
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def process_batch(self, directory: str) -> Dict[str, Tuple[List[str], Dict]]:
        """
        Process all supported documents in a directory (non-recursive).
        
        This method processes every supported file in a single directory and
        collects the results. Useful for batch ingestion workflows.
        
        Args:
            directory (str): Path to directory containing documents
            
        Returns:
            Dict[str, Tuple[List[str], Dict]]: Mapping of:
                - Key: filename (e.g., "document.pdf")
                - Value: (chunks, metadata) tuple from process_document()
                
                If a file fails to process, it's skipped (logged but not included)
        
        Example:
            >>> processor = DocumentProcessor()
            >>> results = processor.process_batch("./data/raw")
            >>> for filename, (chunks, meta) in results.items():
            ...     print(f"{filename}: {meta['num_chunks']} chunks")
            guide.pdf: 42 chunks
            faq.txt: 5 chunks
        """
        results = {}  # Dictionary to store results
        dir_path = Path(directory)  # Convert to Path object
        
        # Iterate through files in the directory
        for file_path in dir_path.iterdir():  # .iterdir() yields all files/dirs
            # Check if it's a file (not a directory) with supported extension
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    # Process this file
                    chunks, metadata = self.process_document(str(file_path))
                    # Store in results dictionary using filename as key
                    results[file_path.name] = (chunks, metadata)
                except Exception as e:
                    # Log the error but continue processing other files
                    logger.error(f"Failed to process {file_path.name}: {e}")
        
        return results
