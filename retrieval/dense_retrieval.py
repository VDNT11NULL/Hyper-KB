"""
Dense retrieval using FAISS for vector similarity search.
Supports both flat (exact) and IVF (approximate) indexes.
"""

import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
from datetime import datetime
from kb.kb_base import DenseRetriever, RetrievalResult, InteractionMetadata
from kb.kb_enhanced import EnhancedKnowledgeBase


class FAISSRetriever(DenseRetriever):
    """
    FAISS-based dense retrieval with multiple index types.
    
    Index Types:
    - 'flat': Exact search, best for <1M vectors
    - 'ivf': Approximate search with IVF, best for >1M vectors
    
    Advantages:
    - Fast similarity search
    - GPU acceleration support
    - Serializable indexes
    - Memory efficient
    
    Best for: Semantic similarity, large-scale retrieval
    """
    
    def __init__(
        self,
        kb: EnhancedKnowledgeBase,
        dimension: int = 384,
        index_type: str = 'flat',
        nlist: int = 100,
        index_path: Optional[str] = None
    ):
        """
        Initialize FAISS retriever.
        
        Args:
            kb: Knowledge base instance
            dimension: Embedding dimension
            index_type: 'flat' or 'ivf'
            nlist: Number of clusters for IVF (only used if index_type='ivf')
            index_path: Path to save/load index file
        """
        self.kb = kb
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.index_path = index_path or f'faiss_index_{index_type}.index'
        
        self.index = None
        self.interaction_ids = []
        
        # Try to load existing index
        if os.path.exists(self.index_path):
            self.load_index()
    
    def _create_index(self, n_vectors: int) -> faiss.Index:
        """Create FAISS index based on type."""
        if self.index_type == 'flat':
            # Exact search using L2 distance
            index = faiss.IndexFlatL2(self.dimension)
            print(f"✓ Created Flat index for exact search")
            
        elif self.index_type == 'ivf':
            # Approximate search using IVF
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            print(f"✓ Created IVF index with {self.nlist} clusters")
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        return index
    
    def index_embeddings(
        self,
        interaction_ids: Optional[List[str]] = None,
        embeddings: Optional[np.ndarray] = None
    ):
        """
        Index embeddings for dense retrieval.
        
        Args:
            interaction_ids: List of interaction IDs (if None, loads all from KB)
            embeddings: Corresponding embeddings (if None, loads from KB)
        """
        # Load from KB if not provided
        if interaction_ids is None or embeddings is None:
            interaction_ids, embeddings = self._load_embeddings_from_kb()
        
        if len(interaction_ids) == 0:
            print("⚠ Warning: No embeddings to index")
            return
        
        # Convert to numpy array and ensure correct shape
        embeddings = np.array(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Validate dimension
        if embeddings.shape[1] != self.dimension:
            print(f"⚠ Warning: Embedding dimension mismatch. "
                  f"Expected {self.dimension}, got {embeddings.shape[1]}")
            self.dimension = embeddings.shape[1]
        
        # Create index
        self.index = self._create_index(len(embeddings))
        
        # Train IVF index if needed
        if self.index_type == 'ivf':
            if len(embeddings) < self.nlist:
                print(f"⚠ Warning: Not enough vectors ({len(embeddings)}) "
                      f"to train IVF with {self.nlist} clusters. Using Flat index instead.")
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                print(f"Training IVF index on {len(embeddings)} vectors...")
                self.index.train(embeddings)
                print("✓ Training complete")
        
        # Add vectors to index
        self.index.add(embeddings)
        self.interaction_ids = interaction_ids
        
        print(f"✓ Indexed {len(interaction_ids)} embeddings")
        
        # Save index
        self.save_index()
    
    def _load_embeddings_from_kb(self) -> Tuple[List[str], np.ndarray]:
        """Load all embeddings from knowledge base."""
        embeddings_docs = list(self.kb.embeddings.find({}))
        
        interaction_ids = []
        embeddings = []
        
        for doc in embeddings_docs:
            interaction_ids.append(doc['interaction_id'])
            embeddings.append(doc['embedding_vector'])
        
        if embeddings:
            embeddings = np.array(embeddings, dtype=np.float32)
        else:
            embeddings = np.array([], dtype=np.float32).reshape(0, self.dimension)
        
        return interaction_ids, embeddings
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Perform dense retrieval using FAISS.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional filters (applied post-retrieval)
            
        Returns:
            List of RetrievalResult objects
        """
        if self.index is None:
            print("⚠ Warning: FAISS not indexed. Indexing now...")
            self.index_embeddings()
        
        if self.index.ntotal == 0:
            print("⚠ Warning: No vectors in index")
            return []
        
        # Ensure query embedding is correct shape
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Convert to results
        results = []
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            interaction_id = self.interaction_ids[idx]
            
            # Get full interaction
            interaction = self.kb.get_interaction_by_id(interaction_id)
            if not interaction:
                continue
            
            # Apply filters if provided
            if filters:
                if 'session_id' in filters:
                    if interaction['session_id'] != filters['session_id']:
                        continue
            
            # Convert distance to similarity score (inverse)
            # Lower distance = higher similarity
            score = float(1.0 / (1.0 + distance))
            
            # Create metadata object
            meta_dict = interaction.get('metadata', {})
            metadata = InteractionMetadata(
                timestamp=meta_dict.get('timestamp', datetime.utcnow()),
                turn_number=meta_dict.get('turn_number', 0),
                session_duration_so_far=meta_dict.get('session_duration_so_far', 0.0),
                session_id=interaction['session_id'],
                previous_interaction_id=meta_dict.get('previous_interaction_id'),
                access_count=meta_dict.get('access_count', 0),
                last_accessed=meta_dict.get('last_accessed'),
                retrieval_scores=meta_dict.get('retrieval_scores', []),
                topic_shift_score=meta_dict.get('topic_shift_score', 0.0),
                context_stability=meta_dict.get('context_stability', 1.0)
            )
            
            result = RetrievalResult(
                interaction_id=interaction_id,
                query_text=interaction['query_text'],
                response_text=interaction['response_text'],
                score=score,
                rank=rank,
                retrieval_method='faiss',
                metadata=metadata,
                features=interaction.get('features')
            )
            results.append(result)
        
        return results
    
    def save_index(self):
        """Save FAISS index to disk."""
        if self.index is not None:
            try:
                faiss.write_index(self.index, self.index_path)
                # Save interaction IDs mapping
                ids_path = self.index_path + '.ids.npy'
                np.save(ids_path, np.array(self.interaction_ids))
                print(f"✓ Saved FAISS index to {self.index_path}")
            except Exception as e:
                print(f"✗ Error saving index: {e}")
    
    def load_index(self):
        """Load FAISS index from disk."""
        try:
            self.index = faiss.read_index(self.index_path)
            # Load interaction IDs mapping
            ids_path = self.index_path + '.ids.npy'
            if os.path.exists(ids_path):
                self.interaction_ids = np.load(ids_path).tolist()
                print(f"✓ Loaded FAISS index from {self.index_path} "
                      f"({self.index.ntotal} vectors)")
            else:
                print(f"⚠ Warning: Index file found but IDs mapping missing")
        except Exception as e:
            print(f"⚠ Warning: Could not load index: {e}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of stored embeddings."""
        return self.dimensions