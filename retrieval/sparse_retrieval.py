"""
Sparse retrieval implementations: BM25 and MongoDB Full-Text Search.
Both approaches implemented for comparison and benchmarking.
"""

from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import numpy as np
from datetime import datetime
from kb.kb_base import SparseRetriever, RetrievalResult, InteractionMetadata
from kb.kb_enhanced import EnhancedKnowledgeBase


class BM25Retriever(SparseRetriever):
    """
    BM25-based sparse retrieval using rank-bm25 library.
    
    Advantages:
    - Tunable parameters (k1, b)
    - Proven ranking formula
    - Fast for small-to-medium corpora
    - No database dependencies
    
    Best for: Controlled experiments, parameter tuning studies
    """
    
    def __init__(
        self,
        kb: EnhancedKnowledgeBase,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            kb: Knowledge base instance
            k1: BM25 term saturation parameter (default: 1.5)
            b: BM25 length normalization parameter (default: 0.75)
        """
        self.kb = kb
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.corpus_ids = []
        self.corpus_metadata = []
    
    def index_documents(self, documents: Optional[List[Dict]] = None):
        """
        Index documents for BM25 retrieval.
        
        Args:
            documents: List of interaction documents (if None, loads all from KB)
        """
        if documents is None:
            # Load all interactions from KB
            documents = list(self.kb.interactions.find({}))
        
        if not documents:
            print("⚠ Warning: No documents to index")
            return
        
        # Prepare corpus
        tokenized_corpus = []
        self.corpus_ids = []
        self.corpus_metadata = []
        
        for doc in documents:
            # Combine query and response for indexing
            text = f"{doc['query_text']} {doc['response_text']}"
            
            # Get features for additional context
            features = self.kb.features.find_one({'interaction_id': doc['interaction_id']})
            if features and features.get('context_passage'):
                text += f" {features['context_passage']}"
            if features and features.get('keywords'):
                text += " " + " ".join(features['keywords'])
            
            # Simple tokenization (split by whitespace and lowercase)
            tokens = text.lower().split()
            tokenized_corpus.append(tokens)
            
            self.corpus_ids.append(doc['interaction_id'])
            self.corpus_metadata.append(doc.get('metadata', {}))
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        print(f"✓ Indexed {len(tokenized_corpus)} documents with BM25")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Perform BM25 retrieval.
        
        Args:
            query: Query text
            top_k: Number of results to return
            filters: Optional filters (e.g., session_id, date range)
            
        Returns:
            List of RetrievalResult objects
        """
        if self.bm25 is None:
            print("⚠ Warning: BM25 not indexed. Indexing now...")
            self.index_documents()
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            interaction_id = self.corpus_ids[idx]
            score = float(scores[idx])
            
            # Skip if score is too low
            if score < 0.01:
                continue
            
            # Get full interaction
            interaction = self.kb.get_interaction_by_id(interaction_id)
            if not interaction:
                continue
            
            # Apply filters if provided
            if filters:
                if 'session_id' in filters:
                    if interaction['session_id'] != filters['session_id']:
                        continue
            
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
                retrieval_method='bm25',
                metadata=metadata,
                features=interaction.get('features')
            )
            results.append(result)
        
        return results


class MongoFTSRetriever(SparseRetriever):
    """
    MongoDB Full-Text Search based sparse retrieval.
    
    Advantages:
    - Native database integration
    - No external dependencies
    - Automatic index management
    - Good for production deployment
    
    Best for: Integrated systems, production environments
    """
    
    def __init__(self, kb: EnhancedKnowledgeBase):
        """
        Initialize MongoDB FTS retriever.
        
        Args:
            kb: Knowledge base instance
        """
        self.kb = kb
    
    def index_documents(self, documents: Optional[List[Dict]] = None):
        """
        Index documents for MongoDB FTS.
        Note: Indexes are already created in EnhancedKnowledgeBase._setup_indexes()
        """
        # MongoDB text indexes are automatically maintained
        print("✓ MongoDB text indexes are active")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Perform MongoDB FTS retrieval.
        
        Args:
            query: Query text
            top_k: Number of results to return
            filters: Optional filters (e.g., session_id)
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            # Build search query
            search_filter = {'$text': {'$search': query}}
            
            # Add additional filters
            if filters:
                search_filter.update(filters)
            
            # Perform text search with score projection
            results_cursor = self.kb.interactions.find(
                search_filter,
                {'score': {'$meta': 'textScore'}}
            ).sort([('score', {'$meta': 'textScore'})]).limit(top_k)
            
            results = []
            for rank, doc in enumerate(results_cursor, 1):
                interaction_id = doc['interaction_id']
                score = doc.get('score', 0.0)
                
                # Get full interaction
                interaction = self.kb.get_interaction_by_id(interaction_id)
                if not interaction:
                    continue
                
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
                    retrieval_method='mongo_fts',
                    metadata=metadata,
                    features=interaction.get('features')
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"✗ Error in MongoDB FTS search: {e}")
            return []