"""
Embedding service module.

This service is responsible for converting text into semantic vector embeddings.
Embeddings are high-dimensional numerical representations of text that capture
meaning and semantic relationships. Texts with similar meanings have similar embeddings.

What are Embeddings?
- Text → Vector: "The cat sat on the mat" → [0.2, -0.5, 0.8, ..., 0.3] (384 numbers)
- Semantic Meaning: Embeddings capture meaning, not just syntax
- Similarity: Similar texts have similar embeddings (dot product or cosine similarity)
- Foundation of RAG: Embeddings allow semantic search in vector databases

Embedding Model Used: Sentence Transformers
- Lightweight and fast (all-MiniLM-L6-v2: 22MB model, produces 384-dim vectors)
- Fine-tuned for semantic similarity (not just token-level similarity)
- Pre-trained on many document pairs, learns what "similar" means
- Works for sentences, paragraphs, documents up to ~512 tokens

Why Sentence Transformers?
- More efficient than BERT alone (uses mean pooling + normalization)
- Creates normalized embeddings (easier to compare with cosine similarity)
- Good balance between speed and quality for RAG systems
- Can handle variable-length inputs efficiently
"""
from typing import List
import logging
import numpy as np

logger = logging.getLogger("docmind")  # Get logger for this module


class EmbeddingService:
    """
    Service for generating text embeddings using Sentence Transformers.
    
    This service manages the embedding model lifecycle:
    - Loading the model from HuggingFace (or cache)
    - Caching it locally to avoid re-downloading
    - Providing methods to embed single texts or batches
    - Calculating similarity between embeddings
    
    The embedding model acts as a "brain" that understands semantic meaning,
    allowing the RAG system to find documents related to a query.
    
    Architecture:
    1. Model is loaded once during initialization
    2. Model is kept in memory (GPU if available, CPU otherwise)
    3. Texts are encoded to embeddings on demand
    4. Embeddings are sent to vector database for storage and search
    
    Example Usage:
        embedding_svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
        query_embedding = embedding_svc.embed_text("What is AI?")
        document_embeddings = embedding_svc.embed_texts(doc_chunks)
        similarity = embedding_svc.similarity(query_embedding, doc_embeddings[0])
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_folder: str = None):
        """
        Initialize the embedding service and load the model.
        
        This method loads the embedding model from HuggingFace (or cache).
        The model is downloaded only once and then cached locally.
        
        Args:
            model_name (str): Name of sentence-transformer model from HuggingFace hub
                Default: "all-MiniLM-L6-v2" (22MB, 384-dim, fast)
                Other options:
                - "all-mpnet-base-v2": Better quality, larger (420MB), 768-dim
                - "sentence-transformers/all-distilroberta-v1": Fast, 768-dim
                - "intfloat/multilingual-e5-base": Multilingual support
            
            cache_folder (str): Where to cache downloaded models
                If None, uses HuggingFace default (~/.cache/huggingface)
                Caching avoids re-downloading the large model file
        
        Instance Variables:
            self.model_name: Stores the model name for reference
            self.cache_folder: Stores cache location
            self.model: The loaded SentenceTransformer instance (set in _load_model)
            self.dimension: Vector dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.model = None  # Will be set by _load_model()
        self.dimension = None  # Will be set by _load_model()
        
        # Load the model (can be slow for first-time download)
        self._load_model()
    
    def _load_model(self):
        """
        Load the embedding model from HuggingFace.
        
        This is called during __init__. It handles:
        1. Importing SentenceTransformer (with error handling)
        2. Downloading/loading the model
        3. Determining the vector dimension
        4. Logging the result
        
        The model is kept in memory after loading for fast inference.
        On first run, this downloads ~22-420MB depending on model.
        On subsequent runs, loads from cache (very fast).
        """
        try:
            # Import SentenceTransformer (from sentence-transformers library)
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Create and load the model
            # This will download from HuggingFace if not cached
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_folder
            )
            
            # ================================================================
            # Determine vector dimension by encoding a test string
            # ================================================================
            # Different models produce different dimensions:
            # - all-MiniLM-L6-v2: 384 dimensions
            # - all-mpnet-base-v2: 768 dimensions
            # We need to know this to create the vector database collection
            dummy_embedding = self.model.encode(["test"])  # Encode dummy text
            self.dimension = dummy_embedding[0].shape[0]  # Get first (and only) embedding's length
            
            logger.info(f"Embedding model loaded. Dimension: {self.dimension}")
        except ImportError:
            error_msg = "sentence-transformers not installed. Install with: pip install sentence-transformers"
            logger.error(error_msg)
            # Don't crash; model remains None, methods will handle it
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            # Don't crash; model remains None, methods will handle it
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        This is a convenience method for embedding one piece of text.
        Internally, it wraps the text in a list and calls the batch method.
        
        Args:
            text (str): Text to convert to embedding
                Can be a word, sentence, paragraph, or short document
                Length: ~1-512 tokens (the model was trained on this range)
            
        Returns:
            np.ndarray: 1D array of floats (embedding vector)
                Shape: (dimension,) e.g., (384,) for all-MiniLM-L6-v2
                Values: typically between -1 and 1 (normalized)
                
        Example:
            >>> service = EmbeddingService()
            >>> embedding = service.embed_text("Hello world")
            >>> print(embedding.shape)
            (384,)
            >>> print(embedding[:5])  # First 5 dimensions
            [ 0.2  -0.5   0.8  -0.1   0.3]
        """
        if self.model is None:
            logger.error("Model not loaded")
            return np.array([])  # Return empty array if model not available
        
        try:
            # encode() returns a numpy array of shape (n_texts, dimension)
            # We wrap text in list: ["text"] → [[embeddings]]
            embedding = self.model.encode([text])
            # Extract first (and only) embedding from the list
            return embedding[0]  # Returns shape (384,)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.array([])  # Return empty array on error
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts (more efficient than one-by-one).
        
        This method is optimized for batch processing:
        - Processes texts in batches (default 32 at a time)
        - Uses GPU if available (much faster than CPU)
        - More efficient than calling embed_text() in a loop
        
        When embedding a 1000-document corpus:
        - One-by-one: 1000 separate forward passes
        - Batched (size 32): 32 forward passes (31x faster!)
        
        Args:
            texts (List[str]): List of texts to embed
                Example: ["FastAPI is great", "I love Python", "Machine learning rocks"]
                Length: 1-million texts is fine (will process in batches)
            
            batch_size (int): How many texts to process at once. Default 32.
                Larger batch = faster but uses more GPU/CPU memory
                Too large batch = OutOfMemory error
                32-64 typically optimal for most models
            
        Returns:
            np.ndarray: 2D array of embeddings
                Shape: (num_texts, dimension)
                Example for 3 texts with 384-dim model: (3, 384)
                Each row is a single text's embedding
                
        Example:
            >>> service = EmbeddingService()
            >>> texts = ["Document 1", "Document 2", "Document 3"]
            >>> embeddings = service.embed_texts(texts, batch_size=32)
            >>> print(embeddings.shape)
            (3, 384)
            >>> similarity = np.dot(embeddings[0], embeddings[1])  # Compare first two
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []  # Return empty list if model not available
        
        try:
            # encode() handles batching internally
            # Returns numpy array of shape (len(texts), dimension)
            embeddings = self.model.encode(texts, batch_size=batch_size)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []  # Return empty list on error
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Cosine similarity measures how "aligned" two vectors are:
        - 1.0 = identical direction (very similar meaning)
        - 0.5 = moderate alignment
        - 0.0 = perpendicular (orthogonal, unrelated)
        - -1.0 = opposite direction (contradictory)
        
        For normalized embeddings (like sentence-transformers),
        cosine similarity is equivalent to dot product.
        
        Why Cosine Similarity?
        - Works in high dimensions (unlike Euclidean distance)
        - Invariant to magnitude (only cares about direction)
        - Fast to compute (just dot product)
        - Matches human intuition of semantic similarity
        
        Args:
            embedding1 (np.ndarray): First embedding vector, shape (dimension,)
            embedding2 (np.ndarray): Second embedding vector, shape (dimension,)
                Both should be 1D arrays of the same size
                Example: both shape (384,) for all-MiniLM-L6-v2
            
        Returns:
            float: Similarity score between -1 and 1
                Typically 0-1 for normalized embeddings
                Use as a threshold: score > 0.5 means relevant
        
        Example:
            >>> service = EmbeddingService()
            >>> emb1 = service.embed_text("The cat sat")
            >>> emb2 = service.embed_text("A feline sits")
            >>> emb3 = service.embed_text("Python programming")
            >>> print(service.similarity(emb1, emb2))  # Similar texts
            0.89
            >>> print(service.similarity(emb1, emb3))  # Different texts
            0.12
        """
        try:
            # Import cosine_similarity from scikit-learn
            from sklearn.metrics.pairwise import cosine_similarity
            
            # cosine_similarity expects 2D arrays (list of vectors)
            # [embedding1] wraps single vector in a list
            # [embedding2] wraps single vector in a list
            # Result: [[similarity_value]] (2D array)
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            # Extract the single float value and return
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0  # Return 0 (no similarity) on error
    
    def get_dimension(self) -> int:
        """
        Get the vector dimension of the embedding model.
        
        This is needed to create the vector database collection.
        The dimension must match the embedding size (384 for default model).
        
        Returns:
            int: Dimension of embeddings, e.g., 384
                Returns None if model not loaded
        
        Example:
            >>> service = EmbeddingService()
            >>> dim = service.get_dimension()
            >>> print(dim)
            384
        """
        return self.dimension
