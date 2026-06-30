"""
Vector database service.
Manages interactions with Qdrant or ChromaDB for vector storage and retrieval.
"""
from typing import List, Dict, Optional, Tuple
import logging
import uuid

logger = logging.getLogger("docmind")


class VectorDBService:
    """Service for managing vector database operations."""
    
    def __init__(self, db_type: str = "qdrant", **kwargs):
        """
        Initialize vector database service.
        
        Args:
            db_type: Type of database (qdrant or chroma)
            **kwargs: Configuration parameters
        """
        self.db_type = db_type.lower()
        self.config = kwargs
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the database client."""
        try:
            if self.db_type == "qdrant":
                self._init_qdrant()
            elif self.db_type == "chroma":
                self._init_chroma()
            else:
                logger.warning(f"Unknown database type: {self.db_type}")
        except Exception as e:
            logger.error(f"Error initializing vector DB client: {e}")
    
    def _init_qdrant(self):
        """Initialize Qdrant client."""
        try:
            from qdrant_client import QdrantClient
            
            url = self.config.get("url", "http://localhost:6333")
            self.client = QdrantClient(url=url)
            logger.info(f"Qdrant client initialized at {url}")
        except ImportError:
            logger.error("qdrant-client not installed. Install with: pip install qdrant-client")
        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
    
    def _init_chroma(self):
        """Initialize ChromaDB client."""
        try:
            import chromadb
            
            self.client = chromadb.Client()
            logger.info("ChromaDB client initialized")
        except ImportError:
            logger.error("chromadb not installed. Install with: pip install chromadb")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
    
    def create_collection(self, collection_name: str, vector_size: int) -> bool:
        """
        Create a new collection in the vector database.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of the vectors
            
        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            logger.error("Vector DB client not initialized")
            return False
        
        try:
            if self.db_type == "qdrant":
                return self._create_qdrant_collection(collection_name, vector_size)
            elif self.db_type == "chroma":
                return self._create_chroma_collection(collection_name)
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    def _create_qdrant_collection(self, collection_name: str, vector_size: int) -> bool:
        """Create Qdrant collection."""
        try:
            from qdrant_client.models import Distance, VectorParams
            
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Qdrant collection created: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {e}")
            return False
    
    def _create_chroma_collection(self, collection_name: str) -> bool:
        """Create ChromaDB collection."""
        try:
            self.client.get_or_create_collection(name=collection_name)
            logger.info(f"ChromaDB collection created: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating ChromaDB collection: {e}")
            return False
    
    def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        texts: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> bool:
        """
        Add vectors to the database.
        
        Args:
            collection_name: Name of the collection
            vectors: List of vector embeddings
            texts: List of text chunks
            metadata: List of metadata dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            logger.error("Vector DB client not initialized")
            return False
        
        if not vectors or not texts:
            logger.warning("Empty vectors or texts")
            return False
        
        try:
            if self.db_type == "qdrant":
                return self._add_qdrant_vectors(collection_name, vectors, texts, metadata)
            elif self.db_type == "chroma":
                return self._add_chroma_vectors(collection_name, vectors, texts, metadata)
        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
            return False
    
    def _add_qdrant_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        texts: List[str],
        metadata: Optional[List[Dict]]
    ) -> bool:
        """Add vectors to Qdrant."""
        try:
            from qdrant_client.models import PointStruct
            
            points = []
            for i, (vector, text) in enumerate(zip(vectors, texts)):
                point_metadata = metadata[i] if metadata else {}
                point_metadata["text"] = text
                
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload=point_metadata
                    )
                )
            
            self.client.upsert(collection_name=collection_name, points=points)
            logger.info(f"Added {len(points)} vectors to {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error adding Qdrant vectors: {e}")
            return False
    
    def _add_chroma_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        texts: List[str],
        metadata: Optional[List[Dict]]
    ) -> bool:
        """Add vectors to ChromaDB."""
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            
            ids = [str(uuid.uuid4()) for _ in texts]
            metadatas = metadata or [{}] * len(texts)
            
            collection.add(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas
            )
            logger.info(f"Added {len(ids)} vectors to {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error adding ChromaDB vectors: {e}")
            return False
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar vectors in the database.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of tuples (text, score, metadata)
        """
        if self.client is None:
            logger.error("Vector DB client not initialized")
            return []
        
        try:
            if self.db_type == "qdrant":
                return self._search_qdrant(collection_name, query_vector, top_k)
            elif self.db_type == "chroma":
                return self._search_chroma(collection_name, query_vector, top_k)
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    def _search_qdrant(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int
    ) -> List[Tuple[str, float, Dict]]:
        """Search in Qdrant."""
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            
            output = []
            for result in results:
                text = result.payload.get("text", "")
                score = result.score
                metadata = {k: v for k, v in result.payload.items() if k != "text"}
                output.append((text, score, metadata))
            
            return output
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return []
    
    def _search_chroma(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int
    ) -> List[Tuple[str, float, Dict]]:
        """Search in ChromaDB."""
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )
            
            output = []
            if results and results['documents']:
                for texts, distances, metadatas in zip(
                    results['documents'],
                    results['distances'],
                    results['metadatas']
                ):
                    for text, distance, metadata in zip(texts, distances, metadatas):
                        # Convert distance to similarity score (1 / (1 + distance))
                        score = 1 / (1 + distance)
                        output.append((text, score, metadata or {}))
            
            return output
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        if self.client is None:
            logger.error("Vector DB client not initialized")
            return False
        
        try:
            if self.db_type == "qdrant":
                self.client.delete_collection(collection_name)
            elif self.db_type == "chroma":
                self.client.delete_collection(name=collection_name)
            
            logger.info(f"Collection deleted: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
