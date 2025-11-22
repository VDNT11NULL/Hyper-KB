"""
Adaptive retrieval strategies for handling conversational drift.
Adjusts retrieval parameters based on detected drift.
"""

from typing import List, Optional, Dict
import numpy as np
from kb.kb_base import RetrievalResult
from .drift_detector import DriftDetector
from .context_tracker import ContextTracker


class AdaptiveRetriever:
    """
    Adapts retrieval strategy based on conversational drift.
    
    Dynamically adjusts:
    - Session vs global search balance
    - Recency weighting
    - Retrieval scope
    """
    
    def __init__(
        self,
        hybrid_retriever,
        drift_detector: Optional[DriftDetector] = None,
        context_tracker: Optional[ContextTracker] = None
    ):
        """
        Initialize adaptive retriever.
        
        Args:
            hybrid_retriever: HybridRetriever instance
            drift_detector: DriftDetector instance
            context_tracker: ContextTracker instance
        """
        self.hybrid_retriever = hybrid_retriever
        self.drift_detector = drift_detector or DriftDetector()
        self.context_tracker = context_tracker or ContextTracker()
    
    def retrieve_adaptive(
        self,
        query: str,
        query_embedding: np.ndarray,
        session_id: str,
        conversation_history: List[Dict],
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Perform adaptive retrieval based on drift state.
        
        Args:
            query: Query text
            query_embedding: Query embedding
            session_id: Session identifier
            conversation_history: Recent conversation history
            top_k: Number of results
            
        Returns:
            Adaptively retrieved and re-ranked results
        """
        # Compute current drift state
        drift_score = self.drift_detector.compute_drift_score(
            query,
            conversation_history
        )
        
        # Get retrieval bias
        bias = self.context_tracker.get_retrieval_bias(session_id)
        
        # Adjust retrieval strategy based on drift
        if drift_score > 0.6:  # High drift - recent topic shift
            results = self._retrieve_with_expansion(
                query,
                query_embedding,
                session_id,
                top_k
            )
        elif drift_score < 0.3:  # Low drift - stable topic
            results = self._retrieve_session_focused(
                query,
                query_embedding,
                session_id,
                top_k
            )
        else:  # Moderate drift - balanced approach
            results = self._retrieve_balanced(
                query,
                query_embedding,
                session_id,
                top_k
            )
        
        # Apply recency reweighting
        results = self._apply_recency_boost(
            results,
            bias['recency_weight']
        )
        
        return results
    
    def _retrieve_with_expansion(
        self,
        query: str,
        query_embedding: np.ndarray,
        session_id: str,
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Retrieve with expanded search (for high drift).
        Prioritizes global search over session context.
        """
        # Get more results globally
        global_results = self.hybrid_retriever.search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k * 2,
            filters=None  # No session filter
        )
        
        # Get some session-specific results
        session_results = self.hybrid_retriever.search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            filters={'session_id': session_id}
        )
        
        # Merge with bias toward global
        return self._merge_results(
            global_results,
            session_results,
            global_weight=0.7,
            top_k=top_k
        )
    
    def _retrieve_session_focused(
        self,
        query: str,
        query_embedding: np.ndarray,
        session_id: str,
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Retrieve focused on session (for low drift).
        Prioritizes session context.
        """
        # Primarily session results
        session_results = self.hybrid_retriever.search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            filters={'session_id': session_id}
        )
        
        # Fill with global if needed
        if len(session_results) < top_k:
            global_results = self.hybrid_retriever.search(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k - len(session_results),
                filters=None
            )
            session_results.extend(global_results)
        
        return session_results[:top_k]
    
    def _retrieve_balanced(
        self,
        query: str,
        query_embedding: np.ndarray,
        session_id: str,
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Balanced retrieval (for moderate drift).
        Equal weight to session and global.
        """
        session_results = self.hybrid_retriever.search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            filters={'session_id': session_id}
        )
        
        global_results = self.hybrid_retriever.search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            filters=None
        )
        
        return self._merge_results(
            session_results,
            global_results,
            global_weight=0.5,
            top_k=top_k
        )
    
    @staticmethod
    def _merge_results(
        results_a: List[RetrievalResult],
        results_b: List[RetrievalResult],
        global_weight: float,
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Merge two result sets with weighting.
        
        Args:
            results_a: First result set
            results_b: Second result set
            global_weight: Weight for second set (0-1)
            top_k: Number to return
            
        Returns:
            Merged and re-ranked results
        """
        # Reweight scores
        for r in results_a:
            r.score *= (1 - global_weight)
        
        for r in results_b:
            r.score *= global_weight
        
        # Combine and deduplicate
        seen = set()
        merged = []
        
        for r in results_a + results_b:
            if r.interaction_id not in seen:
                seen.add(r.interaction_id)
                merged.append(r)
        
        # Re-sort and re-rank
        merged.sort(key=lambda r: r.score, reverse=True)
        for rank, r in enumerate(merged[:top_k], 1):
            r.rank = rank
        
        return merged[:top_k]
    
    @staticmethod
    def _apply_recency_boost(
        results: List[RetrievalResult],
        recency_weight: float
    ) -> List[RetrievalResult]:
        """
        Apply recency boost to result scores.
        
        Args:
            results: Results to boost
            recency_weight: Weight for recency (0-1)
            
        Returns:
            Re-ranked results
        """
        if not results or recency_weight == 0:
            return results
        
        # Find most recent timestamp
        max_timestamp = max(r.metadata.timestamp for r in results)
        
        for result in results:
            # Compute recency factor (0-1, newer = higher)
            time_delta = (max_timestamp - result.metadata.timestamp).total_seconds()
            max_delta = max(
                (max_timestamp - r.metadata.timestamp).total_seconds()
                for r in results
            )
            
            if max_delta > 0:
                recency_factor = 1.0 - (time_delta / max_delta)
            else:
                recency_factor = 1.0
            
            # Blend with original score
            result.score = (
                (1 - recency_weight) * result.score +
                recency_weight * recency_factor
            )
        
        # Re-sort and re-rank
        results.sort(key=lambda r: r.score, reverse=True)
        for rank, r in enumerate(results, 1):
            r.rank = rank
        
        return results