"""
Utility functions for context processing.
Handles deduplication, truncation, and token management.
"""

import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from kb.kb_base import RetrievalResult


class ContextUtils:
    """Utilities for context processing."""
    
    def __init__(self, embedder=None):
        """Initialize with optional embedder for similarity checks."""
        self.embedder = embedder or SentenceTransformer("all-MiniLM-L6-v2")
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count (rough approximation: 1 token ~ 4 chars)."""
        return len(text) // 4
    
    def deduplicate_contexts(
        self,
        results: List[RetrievalResult],
        similarity_threshold: float = 0.85
    ) -> List[RetrievalResult]:
        """
        Remove near-duplicate contexts based on semantic similarity.
        
        Args:
            results: List of retrieval results
            similarity_threshold: Cosine similarity threshold for duplicates
            
        Returns:
            Deduplicated list of results
        """
        if len(results) <= 1:
            return results
        
        # Extract texts and compute embeddings
        texts = [r.response_text for r in results]
        embeddings = self.embedder.encode(texts)
        
        # Compute pairwise similarities
        keep_mask = [True] * len(results)
        
        for i in range(len(results)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(results)):
                if not keep_mask[j]:
                    continue
                
                # Cosine similarity
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                
                if sim > similarity_threshold:
                    # Keep higher-ranked result
                    if results[i].rank < results[j].rank:
                        keep_mask[j] = False
                    else:
                        keep_mask[i] = False
                        break
        
        return [r for r, keep in zip(results, keep_mask) if keep]
    
    @staticmethod
    def truncate_to_tokens(
        text: str,
        max_tokens: int,
        suffix: str = "..."
    ) -> str:
        """
        Truncate text to approximate token limit.
        
        Args:
            text: Input text
            max_tokens: Maximum token count
            suffix: Suffix to add when truncated
            
        Returns:
            Truncated text
        """
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars - len(suffix)] + suffix
    
    @staticmethod
    def extract_most_relevant(
        results: List[RetrievalResult],
        max_contexts: int
    ) -> List[RetrievalResult]:
        """Extract top-K most relevant results."""
        return sorted(results, key=lambda r: r.score, reverse=True)[:max_contexts]