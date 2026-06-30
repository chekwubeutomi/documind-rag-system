"""
Embedding service.
Manages embedding model loading and inference using sentence-transformers.
"""
from typing import List
import logging
import numpy as np

logger = logging.getLogger("docmind")


class EmbeddingService:
    """Service for generating embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_folder: str = None):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformer model
            cache_folder: Folder to cache the model
        """
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.model = None
        self.dimension = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_folder
            )
            
            # Get dimension by encoding a dummy text
            dummy_embedding = self.model.encode(["test"])
            self.dimension = dummy_embedding[0].shape[0]
            
            logger.info(f"Embedding model loaded. Dimension: {self.dimension}")
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if self.model is None:
            logger.error("Model not loaded")
            return np.array([])
        
        try:
            embedding = self.model.encode([text])
            return embedding[0]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.array([])
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            
        Returns:
            List of embedding vectors
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        try:
            embeddings = self.model.encode(texts, batch_size=batch_size)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.dimension
