"""
Vector database service module.

Vector databases are specialized databases designed for storing and searching
high-dimensional vectors (embeddings). They're the core of RAG systems!

What is a Vector Database?
Think of it like a regular database but instead of searching by exact match
or keywords, it searches by "semantic similarity":
- Regular DB: "Find where name = 'John'"
- Vector DB: "Find documents semantically similar to this embedding"

Why Vector Databases?
- Traditional search: "AI" matches "artificial" but misses "machine learning"
- Vector search: Understands that AI, artificial intelligence, and machine learning
  are semantically similar even without matching keywords
- Speed: HNSW (Hierarchical Navigable Small World) algorithms find top-K results
  in thousands of vectors in milliseconds
- Scalability: Can handle millions of vectors efficiently

How Vector Search Works (Simplified):
1. Convert query to embedding (using same model as documents)
2. Traverse HNSW graph to find nearby vectors
3. Return top-K closest vectors (by cosine similarity)
4. The "closeness" reflects semantic relevance!

┌──────────────────────────────────────────────────────────┐
│ Vector Space (simplified 2D for visualization)           │
├──────────────────────────────────────────────────────────┤
│                        Topic: Weather                    │
│                             *                            │
│                          rain                            │
│                             *                            │
│             cloud *              * snow                  │
│                                                          │
│     *sunshine      ← Query: "rainy day"                  │
│    (far away)      Returns: rain, cloud, snow (near)    │
│                                                          │
│                  Deep space tech *                       │
│                  (far - different topic)                 │
└──────────────────────────────────────────────────────────┘

Supported Databases:
1. Qdrant - Modern, production-ready
   - Protocol buffer storage (compact, fast)
   - Advanced indexing (HNSW, scalar quantization)
   - API-based (runs as separate service)
   - Best for: Production systems, large scale

2. ChromaDB - Lightweight, developer-friendly
   - In-memory or local file storage
   - Simple API
   - Embedded in Python application
   - Best for: Development, testing, small projects

Database Comparison:
┌────────────┬────────────────┬──────────────┬─────────────┐
│ Feature    │ Qdrant         │ ChromaDB      │ Notes       │
├────────────┼────────────────┼──────────────┼─────────────┤
│ Scale      │ Millions       │ Thousands    │ Qdrant wins │
│ Speed      │ <10ms          │ <50ms        │ Qdrant wins │
│ Setup      │ Docker needed  │ pip install  │ ChromaDB    │
│ Persistence│ Disk (default) │ SQLite       │ Both good   │
│ Deployment │ Service        │ Embedded     │ Different   │
└────────────┴────────────────┴──────────────┴─────────────┘

Key Concepts:
- Collection: Like a table, stores vectors for one type of data
- Point/Document: A vector with metadata (the actual embedding)
- Payload: Metadata attached to each vector (text, source, etc.)
- Similarity Score: 0-1 (1=perfect match, 0=completely different)
"""
from typing import List, Dict, Optional, Tuple
import logging
import uuid

logger = logging.getLogger("docmind")  # Get logger for this module


class VectorDBService:
    """
    Unified service for managing vector database operations.
    
    This service abstracts away the differences between Qdrant and ChromaDB,
    providing a consistent interface for vector storage and retrieval.
    The strategy pattern is used: each database has its own implementation,
    but both implement the same public interface (create_collection, add_vectors, search).
    
    Workflow:
    1. Create service: VectorDBService(db_type="qdrant", url="http://localhost:6333")
    2. Create collection: service.create_collection("documents", vector_size=384)
    3. Add vectors: service.add_vectors("documents", embeddings, texts, metadata)
    4. Search: results = service.search("documents", query_embedding, top_k=5)
    5. Delete: service.delete_collection("documents")
    
    Data Flow in RAG:
    Documents → Embedding Service → Vector embeddings (384-dim floats)
                                         ↓
                              Vector DB Service
                                         ↓
    Query → Embedding Service → Query embedding (same 384-dim)
                                         ↓
                           Vector DB searches for similarity
                                         ↓
                           Returns top-K similar documents
                                         ↓
                              LLM generates answer
    
    Example Usage:
        # Initialize with Qdrant
        db = VectorDBService(db_type="qdrant", url="http://localhost:6333")
        
        # Create a collection for documents (384-dim for all-MiniLM model)
        db.create_collection("faqs", vector_size=384)
        
        # Add documents with embeddings
        embeddings = [[0.2, -0.5, ...], [...], ...]  # 384 floats each
        texts = ["FAQ 1 content", "FAQ 2 content", ...]
        metadata = [{"source": "faq.txt", "page": 1}, ...]
        db.add_vectors("faqs", embeddings, texts, metadata)
        
        # Search for similar documents
        query_embedding = [[0.1, -0.4, ...]]  # 384 floats
        results = db.search("faqs", query_embedding, top_k=5)
        for text, score, meta in results:
            print(f"Relevance: {score:.2f} - {text[:50]}...")
    """
    
    def __init__(self, db_type: str = "qdrant", **kwargs):
        """
        Initialize vector database service with specified backend.
        
        Args:
            db_type (str): Which vector database to use
                Options: "qdrant" (recommended), "chroma"
                Default: "qdrant"
            
            **kwargs: Database-specific configuration
                For Qdrant:
                    url (str): URL to Qdrant server, e.g., "http://localhost:6333"
                    Default: "http://localhost:6333"
                
                For ChromaDB:
                    No special config needed, runs embedded in Python
                
                Examples:
                VectorDBService("qdrant", url="http://localhost:6333")
                VectorDBService("chroma")
        
        Instance Variables:
            self.db_type: Which backend is being used (normalized to lowercase)
            self.config: Dict of kwargs for database configuration
            self.client: The initialized database client (set by _initialize_client)
        """
        self.db_type = db_type.lower()  # Normalize to lowercase ("Qdrant" → "qdrant")
        self.config = kwargs  # Store all kwargs for later use
        self.client = None  # Will be set by _initialize_client()
        
        # Initialize the appropriate database client
        self._initialize_client()
    
    def _initialize_client(self):
        """
        Dispatch initialization to the appropriate database backend.
        
        This acts as a router: based on self.db_type, it calls the right
        initialization method. This pattern makes it easy to add new backends.
        
        Error handling: If initialization fails, self.client remains None.
        Later methods check for this and return False/empty instead of crashing.
        """
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
        """
        Initialize Qdrant client.
        
        Qdrant is a high-performance vector search engine running as a service.
        
        Setup:
        1. Install qdrant-client: pip install qdrant-client
        2. Start Qdrant server: docker run -p 6333:6333 qdrant/qdrant
        3. Creates QdrantClient pointing to the server
        
        Connection:
        - QdrantClient connects to a running Qdrant server via HTTP REST API
        - Default: localhost:6333 (local development)
        - Production: Can connect to remote Qdrant instance
        
        Why Qdrant?
        - HNSW indexing: Extremely fast similarity search
        - Scalar quantization: Compress vectors to save memory (4x compression)
        - Advanced filtering: Filter results by metadata
        - Persistence: Data saved to disk, survives restarts
        """
        try:
            from qdrant_client import QdrantClient
            
            # Get server URL from config (default to localhost for development)
            url = self.config.get("url", "http://localhost:6333")
            
            # Create client that connects to the Qdrant server
            self.client = QdrantClient(url=url)
            logger.info(f"Qdrant client initialized at {url}")
        except ImportError:
            logger.error("qdrant-client not installed. Install with: pip install qdrant-client")
        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
    
    def _init_chroma(self):
        """
        Initialize ChromaDB client.
        
        ChromaDB is a lightweight, embeddable vector database.
        
        Setup:
        1. Install chromadb: pip install chromadb
        2. Create ChromaDB client (no external service needed!)
        3. Data stored in local SQLite database
        
        Connection:
        - ChromaDB runs inside your Python application
        - No separate server process needed
        - Great for development and testing
        - Can also run data in memory (faster but not persistent)
        
        Why ChromaDB?
        - Simple API: just Client() and you're ready
        - No Docker: perfect for local development
        - Good for prototyping and testing
        - Slower than Qdrant but good enough for small projects
        """
        try:
            import chromadb
            
            # Create the ChromaDB client (runs embedded)
            self.client = chromadb.Client()
            logger.info("ChromaDB client initialized")
        except ImportError:
            logger.error("chromadb not installed. Install with: pip install chromadb")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
    
    def create_collection(self, collection_name: str, vector_size: int) -> bool:
        """
        Create a new collection (like a table) in the vector database.
        
        A collection is a container for related vectors. Example:
        - "documents" collection: all document embeddings (384-dim each)
        - "faq" collection: all FAQ embeddings (384-dim each)
        
        Important:
        - Collections are schema-free (no predefined schema needed)
        - Vector size MUST match your embedding model dimension (384 for default)
        - Must be created before adding vectors
        - Names must be unique within a database
        
        Args:
            collection_name (str): Name for the collection
                Example: "documents", "faqs", "knowledge_base"
                Naming: lowercase with underscores recommended
            
            vector_size (int): Dimension of vectors to store
                Must match the dimension of your embedding model:
                - all-MiniLM-L6-v2: 384 dimensions
                - all-mpnet-base-v2: 768 dimensions
                - Mismatch will cause errors when adding vectors!
        
        Returns:
            bool: True if collection created successfully, False otherwise
            
        Example:
            >>> db = VectorDBService("qdrant")
            >>> success = db.create_collection("documents", vector_size=384)
            >>> if success:
            ...     print("Ready to store vectors!")
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
        """
        Create Qdrant collection with HNSW indexing.
        
        Qdrant automatically creates HNSW (Hierarchical Navigable Small World)
        indexes for fast similarity search.
        
        Distance Metric - Cosine Similarity:
        - Measures angle between vectors (not distance)
        - 1.0 = identical vectors
        - 0.0 = perpendicular (unrelated)
        - -1.0 = opposite vectors
        For normalized embeddings (sentence-transformers), this is perfect!
        
        Args:
            collection_name: Name of collection to create
            vector_size: Dimension of vectors (e.g., 384)
        
        Returns:
            bool: True if successful
        """
        try:
            from qdrant_client.models import Distance, VectorParams
            
            # Recreate (delete if exists, then create)
            # This ensures clean state if collection already exists
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,  # Each vector is 384-dimensional
                    distance=Distance.COSINE  # Use cosine distance for similarity
                )
            )
            logger.info(f"Qdrant collection created: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {e}")
            return False
    
    def _create_chroma_collection(self, collection_name: str) -> bool:
        """
        Create ChromaDB collection.
        
        ChromaDB is more flexible - doesn't require specifying vector size upfront.
        It infers the dimension from the first vectors added.
        
        Args:
            collection_name: Name of collection to create
        
        Returns:
            bool: True if successful
        """
        try:
            # get_or_create_collection: gets if exists, creates if not
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
        Add vectors (with associated text and metadata) to a collection.
        
        This is where you store document embeddings for later retrieval.
        
        Data Structure:
        - Each document is split into chunks
        - Each chunk is embedded to a vector (384 floats)
        - Vector is stored with original text and metadata
        
        Example storage:
        ┌─────────────────────────────────────────────┐
        │ Vector (384 floats)                         │
        │ [0.234, -0.891, 0.123, ..., 0.456]        │
        ├─────────────────────────────────────────────┤
        │ Payload (Metadata):                         │
        │ - text: "Machine learning is a subset of AI"│
        │ - source: "ML_guide.pdf"                    │
        │ - page: 5                                   │
        │ - chunk_index: 12                           │
        └─────────────────────────────────────────────┘
        
        Args:
            collection_name (str): Which collection to add to
                Must exist (create with create_collection first)
            
            vectors (List[List[float]]): List of embedding vectors
                Each vector is a list of floats (384 for default model)
                Length must match texts length
                Example: [[0.1, 0.2, ...], [0.3, 0.4, ...]]
            
            texts (List[str]): Corresponding text chunks for each vector
                These are the original document snippets
                Returned in search results
                Example: ["First chunk", "Second chunk"]
            
            metadata (Optional[List[Dict]]): Optional metadata for each vector
                Each dict can contain any key-value pairs
                Common: {source, page, chunk_index, timestamp}
                If None, defaults to empty dicts
                Example: [{"source": "doc.pdf", "page": 1}, ...]
        
        Returns:
            bool: True if all vectors added successfully, False otherwise
        
        Example:
            >>> embeddings = np.random.rand(10, 384).tolist()  # 10 vectors
            >>> texts = ["Text 1", "Text 2", ...]
            >>> meta = [{"source": "docs.pdf"}, ...]
            >>> success = db.add_vectors("docs", embeddings, texts, meta)
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
        """
        Add vectors to Qdrant collection.
        
        Qdrant uses a "Point" structure:
        - id: Unique identifier (UUID for randomness)
        - vector: The embedding vector
        - payload: Dictionary with metadata (text, source, etc.)
        
        Args:
            Same as add_vectors()
        
        Returns:
            bool: True if successful
        """
        try:
            from qdrant_client.models import PointStruct
            
            points = []  # List of Point objects to upsert
            
            # Build a Point for each vector
            for i, (vector, text) in enumerate(zip(vectors, texts)):
                # Get metadata for this vector (or empty dict if not provided)
                point_metadata = metadata[i] if metadata else {}
                
                # Add the text to metadata (required for search results)
                point_metadata["text"] = text
                
                # Create a Point: unique ID + vector + metadata
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),  # Generate unique ID for this point
                        vector=vector,  # The embedding (384 floats)
                        payload=point_metadata  # Text + other metadata
                    )
                )
            
            # Upsert: insert or update all points atomically
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
        """
        Add vectors to ChromaDB collection.
        
        ChromaDB API:
        - ids: Unique identifier for each document (UUID)
        - embeddings: The vectors (stored as-is)
        - documents: The text (ChromaDB stores this)
        - metadatas: Additional metadata dictionaries
        
        Args:
            Same as add_vectors()
        
        Returns:
            bool: True if successful
        """
        try:
            # Get or create the collection (ChromaDB is flexible)
            collection = self.client.get_or_create_collection(name=collection_name)
            
            # Generate unique IDs for each vector
            ids = [str(uuid.uuid4()) for _ in texts]
            
            # Default to empty metadata dicts if not provided
            metadatas = metadata or [{}] * len(texts)
            
            # Add all vectors to the collection
            collection.add(
                ids=ids,  # Unique identifiers
                embeddings=vectors,  # The vectors
                documents=texts,  # Original text
                metadatas=metadatas  # Metadata dicts
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
        Search for vectors most similar to the query vector.
        
        This is the core of RAG retrieval! Given a query embedding,
        find the K most similar document embeddings.
        
        Search Algorithm (HNSW in Qdrant):
        1. Start at a random node in the HNSW graph
        2. Greedily move to closer neighbors
        3. When at local minimum, jump up a layer
        4. Repeat until you find the K nearest neighbors
        Result: Top-K similar documents in O(log N) time!
        
        Args:
            collection_name (str): Which collection to search
            
            query_vector (List[float]): Query embedding to find similar to
                Must be same dimension as collection (384 for default)
                Should be embedding of user's question
                Example: embedding_service.embed_text("What is AI?")
            
            top_k (int): How many results to return. Default 5.
                More results = more context for LLM, slower processing
                Less results = faster but might miss relevant docs
                Typical: 3-10 documents for RAG
        
        Returns:
            List[Tuple[str, float, Dict]]: List of (text, similarity_score, metadata)
                Sorted by similarity score (highest first)
                - text: Original document chunk
                - similarity_score: 0-1, higher = more similar
                - metadata: Original metadata dict (source, page, etc.)
                
                Empty list if no results or error
        
        Example:
            >>> query_embedding = embedding_svc.embed_text("How does RAG work?")
            >>> results = db.search("documents", query_embedding, top_k=5)
            >>> for text, score, meta in results:
            ...     print(f"{score:.2f}: {text[:50]}... (from {meta['source']})")
            0.89: RAG retrieves relevant documents and uses them... (from guide.pdf)
            0.85: The retrieval step is critical for RAG performance... (from blog.md)
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
        """
        Search in Qdrant collection.
        
        Uses HNSW index for fast similarity search.
        
        Args:
            Same as search()
        
        Returns:
            Tuples of (text, similarity_score, metadata)
        """
        try:
            # Query the vector database (HNSW search)
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,  # What we're looking for
                limit=top_k  # Return top K results
            )
            
            output = []
            # Extract text, score, and metadata from results
            for result in results:
                text = result.payload.get("text", "")  # Get text from payload
                score = result.score  # Similarity score (0-1)
                # Get metadata (everything except the text)
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
        """
        Search in ChromaDB collection.
        
        ChromaDB uses L2 (Euclidean) distance by default.
        We convert to similarity: score = 1 / (1 + distance)
        
        Args:
            Same as search()
        
        Returns:
            Tuples of (text, similarity_score, metadata)
        """
        try:
            # Get the collection
            collection = self.client.get_or_create_collection(name=collection_name)
            
            # Query the collection
            results = collection.query(
                query_embeddings=[query_vector],  # Wrap in list (ChromaDB expects batch)
                n_results=top_k  # Return top K results
            )
            
            output = []
            # Results format: documents, distances, metadatas (all per query)
            if results and results['documents']:
                for texts, distances, metadatas in zip(
                    results['documents'],  # List of document text lists
                    results['distances'],  # List of distance lists
                    results['metadatas']   # List of metadata dict lists
                ):
                    # Iterate through all results for this query
                    for text, distance, metadata in zip(texts, distances, metadatas):
                        # Convert L2 distance to similarity score
                        # distance = 0 → score = 1 (identical)
                        # distance = ∞ → score → 0 (different)
                        score = 1 / (1 + distance)
                        output.append((text, score, metadata or {}))
            
            return output
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete an entire collection.
        
        Useful for cleaning up test data or starting fresh.
        
        Args:
            collection_name (str): Which collection to delete
        
        Returns:
            bool: True if deleted successfully, False otherwise
        """
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
