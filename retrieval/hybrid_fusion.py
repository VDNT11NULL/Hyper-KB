"""
Hybrid fusion strategies for combining sparse and dense retrieval results.
Implements multiple fusion approaches for comparison and benchmarking.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict
from datetime import datetime
from .sparse_retrieval import SparseRetriever
from .dense_retrieval import DenseRetriever
from kb.kb_base import RetrievalResult


class HybridFusion:
    """
    Base class for hybrid fusion strategies.
    Combines results from sparse and dense retrievers.
    """
    
    def __init__(
        self,
        sparse_retriever: SparseRetriever,
        dense_retriever: DenseRetriever
    ):
        """
        Initialize hybrid fusion.
        
        Args:
            sparse_retriever: Sparse retrieval instance
            dense_retriever: Dense retrieval instance
        """
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
    
    def fuse(
        self,
        sparse_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Fuse sparse and dense results.
        Must be implemented by subclasses.
        
        Args:
            sparse_results: Results from sparse retrieval
            dense_results: Results from dense retrieval
            top_k: Number of final results to return
            
        Returns:
            Fused and re-ranked results
        """
        raise NotImplementedError("Subclasses must implement fuse()")


class ReciprocalRankFusion(HybridFusion):
    """
    Reciprocal Rank Fusion (RRF) for combining retrieval results.
    
    RRF Score = sum(1 / (k + rank_i)) for all rankings where doc appears
    
    Advantages:
    - No parameter tuning required
    - Robust to score scale differences
    - Simple and effective
    
    Reference: Cormack et al. (2009) "Reciprocal rank fusion outperforms condorcet 
    and individual rank learning methods"
    """
    
    def __init__(
        self,
        sparse_retriever: SparseRetriever,
        dense_retriever: DenseRetriever,
        k: int = 60
    ):
        """
        Initialize RRF fusion.
        
        Args:
            sparse_retriever: Sparse retrieval instance
            dense_retriever: Dense retrieval instance
            k: Constant for RRF formula (default: 60, from original paper)
        """
        super().__init__(sparse_retriever, dense_retriever)
        self.k = k
    
    def fuse(
        self,
        sparse_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Fuse using Reciprocal Rank Fusion.
        
        Args:
            sparse_results: Results from sparse retrieval
            dense_results: Results from dense retrieval
            top_k: Number of final results to return
            
        Returns:
            Fused results sorted by RRF score
        """
        # Build RRF scores
        rrf_scores = defaultdict(float)
        interaction_map = {}
        
        # Add sparse results
        for result in sparse_results:
            rrf_scores[result.interaction_id] += 1.0 / (self.k + result.rank)
            if result.interaction_id not in interaction_map:
                interaction_map[result.interaction_id] = result
        
        # Add dense results
        for result in dense_results:
            rrf_scores[result.interaction_id] += 1.0 / (self.k + result.rank)
            if result.interaction_id not in interaction_map:
                interaction_map[result.interaction_id] = result
        
        # Sort by RRF score
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )[:top_k]
        
        # Build final results
        fused_results = []
        for rank, interaction_id in enumerate(sorted_ids, 1):
            result = interaction_map[interaction_id]
            result.score = rrf_scores[interaction_id]
            result.rank = rank
            result.retrieval_method = 'hybrid_rrf'
            fused_results.append(result)
        
        return fused_results


class WeightedFusion(HybridFusion):
    """
    Weighted linear combination of sparse and dense scores.
    
    Final Score = alpha * sparse_score + (1 - alpha) * dense_score
    
    Advantages:
    - Tunable weight parameter
    - Simple interpretation
    - Can favor one retrieval method
    
    Best for: When one retrieval method is known to be more reliable
    """
    
    def __init__(
        self,
        sparse_retriever: SparseRetriever,
        dense_retriever: DenseRetriever,
        alpha: float = 0.5,
        normalize: bool = True
    ):
        """
        Initialize weighted fusion.
        
        Args:
            sparse_retriever: Sparse retrieval instance
            dense_retriever: Dense retrieval instance
            alpha: Weight for sparse scores (1-alpha for dense)
            normalize: Whether to normalize scores before fusion
        """
        super().__init__(sparse_retriever, dense_retriever)
        self.alpha = alpha
        self.normalize = normalize
    
    def _normalize_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Min-max normalization of scores."""
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score - min_score < 1e-10:
            # All scores are the same
            for result in results:
                result.score = 1.0
        else:
            for result in results:
                result.score = (result.score - min_score) / (max_score - min_score)
        
        return results
    
    def fuse(
        self,
        sparse_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Fuse using weighted linear combination.
        
        Args:
            sparse_results: Results from sparse retrieval
            dense_results: Results from dense retrieval
            top_k: Number of final results to return
            
        Returns:
            Fused results sorted by weighted score
        """
        # Normalize scores if requested
        if self.normalize:
            sparse_results = self._normalize_scores(sparse_results)
            dense_results = self._normalize_scores(dense_results)
        
        # Build score maps
        sparse_scores = {r.interaction_id: r.score for r in sparse_results}
        dense_scores = {r.interaction_id: r.score for r in dense_results}
        
        # Combine all unique interaction IDs
        all_ids = set(sparse_scores.keys()) | set(dense_scores.keys())
        
        # Build interaction map
        interaction_map = {}
        for result in sparse_results + dense_results:
            if result.interaction_id not in interaction_map:
                interaction_map[result.interaction_id] = result
        
        # Calculate weighted scores
        weighted_scores = {}
        for interaction_id in all_ids:
            sparse_score = sparse_scores.get(interaction_id, 0.0)
            dense_score = dense_scores.get(interaction_id, 0.0)
            weighted_scores[interaction_id] = (
                self.alpha * sparse_score + (1 - self.alpha) * dense_score
            )
        
        # Sort by weighted score
        sorted_ids = sorted(
            weighted_scores.keys(),
            key=lambda x: weighted_scores[x],
            reverse=True
        )[:top_k]
        
        # Build final results
        fused_results = []
        for rank, interaction_id in enumerate(sorted_ids, 1):
            result = interaction_map[interaction_id]
            result.score = weighted_scores[interaction_id]
            result.rank = rank
            result.retrieval_method = f'hybrid_weighted_alpha{self.alpha}'
            fused_results.append(result)
        
        return fused_results


class DistributionBasedFusion(HybridFusion):
    """
    Fusion based on score distribution characteristics.
    
    Normalizes using z-scores and combines based on statistical properties.
    
    Advantages:
    - Handles different score distributions
    - More robust than simple normalization
    - Can detect outliers
    
    Best for: When sparse and dense retrievers have very different score ranges
    """
    
    def __init__(
        self,
        sparse_retriever: SparseRetriever,
        dense_retriever: DenseRetriever,
        alpha: float = 0.5
    ):
        """
        Initialize distribution-based fusion.
        
        Args:
            sparse_retriever: Sparse retrieval instance
            dense_retriever: Dense retrieval instance
            alpha: Weight for sparse scores (1-alpha for dense)
        """
        super().__init__(sparse_retriever, dense_retriever)
        self.alpha = alpha
    
    def _zscore_normalize(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Z-score normalization of scores."""
        if not results:
            return results
        
        scores = np.array([r.score for r in results])
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if std_score < 1e-10:
            # All scores are the same
            for result in results:
                result.score = 0.0
        else:
            normalized = (scores - mean_score) / std_score
            for result, norm_score in zip(results, normalized):
                result.score = float(norm_score)
        
        return results
    
    def fuse(
        self,
        sparse_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Fuse using distribution-based normalization.
        
        Args:
            sparse_results: Results from sparse retrieval
            dense_results: Results from dense retrieval
            top_k: Number of final results to return
            
        Returns:
            Fused results sorted by combined z-score
        """
        # Z-score normalize
        sparse_results = self._zscore_normalize(sparse_results)
        dense_results = self._zscore_normalize(dense_results)
        
        # Build score maps
        sparse_scores = {r.interaction_id: r.score for r in sparse_results}
        dense_scores = {r.interaction_id: r.score for r in dense_results}
        
        # Combine all unique interaction IDs
        all_ids = set(sparse_scores.keys()) | set(dense_scores.keys())
        
        # Build interaction map
        interaction_map = {}
        for result in sparse_results + dense_results:
            if result.interaction_id not in interaction_map:
                interaction_map[result.interaction_id] = result
        
        # Calculate combined scores
        combined_scores = {}
        for interaction_id in all_ids:
            sparse_score = sparse_scores.get(interaction_id, -3.0)  # Low z-score if missing
            dense_score = dense_scores.get(interaction_id, -3.0)
            combined_scores[interaction_id] = (
                self.alpha * sparse_score + (1 - self.alpha) * dense_score
            )
        
        # Sort by combined score
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True
        )[:top_k]
        
        # Build final results
        fused_results = []
        for rank, interaction_id in enumerate(sorted_ids, 1):
            result = interaction_map[interaction_id]
            result.score = combined_scores[interaction_id]
            result.rank = rank
            result.retrieval_method = f'hybrid_distribution_alpha{self.alpha}'
            fused_results.append(result)
        
        return fused_results


class HybridRetriever:
    """
    Unified interface for hybrid retrieval with multiple fusion strategies.
    
    This class provides a simple API for performing hybrid retrieval
    with different fusion methods.
    """
    
    def __init__(
        self,
        sparse_retriever: SparseRetriever,
        dense_retriever: DenseRetriever,
        fusion_method: str = 'rrf',
        fusion_params: Optional[Dict] = None
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            sparse_retriever: Sparse retrieval instance
            dense_retriever: Dense retrieval instance
            fusion_method: 'rrf', 'weighted', or 'distribution'
            fusion_params: Additional parameters for fusion method
        """
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.fusion_method = fusion_method
        self.fusion_params = fusion_params or {}
        
        # Initialize fusion strategy
        self.fusion_strategy = self._create_fusion_strategy()
    
    def _create_fusion_strategy(self) -> HybridFusion:
        """Create fusion strategy based on method name."""
        if self.fusion_method == 'rrf':
            return ReciprocalRankFusion(
                self.sparse_retriever,
                self.dense_retriever,
                k=self.fusion_params.get('k', 60)
            )
        elif self.fusion_method == 'weighted':
            return WeightedFusion(
                self.sparse_retriever,
                self.dense_retriever,
                alpha=self.fusion_params.get('alpha', 0.5),
                normalize=self.fusion_params.get('normalize', True)
            )
        elif self.fusion_method == 'distribution':
            return DistributionBasedFusion(
                self.sparse_retriever,
                self.dense_retriever,
                alpha=self.fusion_params.get('alpha', 0.5)
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        sparse_k: Optional[int] = None,
        dense_k: Optional[int] = None,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval.
        
        Args:
            query: Query text for sparse retrieval
            query_embedding: Query embedding for dense retrieval
            top_k: Number of final results to return
            sparse_k: Number of results from sparse retrieval (default: 2*top_k)
            dense_k: Number of results from dense retrieval (default: 2*top_k)
            filters: Optional filters to apply
            
        Returns:
            Fused and re-ranked results
        """
        # Default to retrieving 2x top_k from each method
        sparse_k = sparse_k or (2 * top_k)
        dense_k = dense_k or (2 * top_k)
        
        # Perform sparse retrieval
        sparse_results = self.sparse_retriever.search(
            query=query,
            top_k=sparse_k,
            filters=filters
        )
        
        # Perform dense retrieval
        dense_results = self.dense_retriever.search(
            query_embedding=query_embedding,
            top_k=dense_k,
            filters=filters
        )
        
        # Fuse results
        fused_results = self.fusion_strategy.fuse(
            sparse_results=sparse_results,
            dense_results=dense_results,
            top_k=top_k
        )
        
        return fused_results
    
    def search_sparse_only(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """Perform sparse-only retrieval."""
        return self.sparse_retriever.search(query, top_k, filters)
    
    def search_dense_only(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """Perform dense-only retrieval."""
        return self.dense_retriever.search(query_embedding, top_k, filters)