"""
Metrics for evaluating prompt enhancement quality.
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer


class PromptEnhancementMetrics:
    """Metrics for prompt quality evaluation."""
    
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    def context_relevance_score(
        self,
        query: str,
        context: str
    ) -> float:
        """
        Measure semantic relevance between query and context.
        
        Returns:
            Cosine similarity score (0-1)
        """
        query_emb = self.embedder.encode([query])[0]
        context_emb = self.embedder.encode([context])[0]
        
        sim = np.dot(query_emb, context_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(context_emb)
        )
        
        return float(sim)
    
    @staticmethod
    def redundancy_score(contexts: List[str]) -> float:
        """
        Measure redundancy in contexts (lower is better).
        
        Returns:
            Redundancy score (0-1, lower is better)
        """
        if len(contexts) <= 1:
            return 0.0
        
        # Simple token overlap measure
        all_tokens = set()
        overlaps = []
        
        for context in contexts:
            tokens = set(context.lower().split())
            if all_tokens:
                overlap = len(tokens & all_tokens) / len(tokens | all_tokens)
                overlaps.append(overlap)
            all_tokens.update(tokens)
        
        return np.mean(overlaps) if overlaps else 0.0
    
    @staticmethod
    def token_efficiency(
        context: str,
        query: str,
        response: str
    ) -> Dict[str, float]:
        """
        Measure token usage efficiency.
        
        Returns:
            Dict with context_ratio and total_tokens
        """
        def count_tokens(text):
            return len(text) // 4
        
        context_tokens = count_tokens(context)
        query_tokens = count_tokens(query)
        response_tokens = count_tokens(response)
        total = context_tokens + query_tokens + response_tokens
        
        return {
            'context_tokens': context_tokens,
            'query_tokens': query_tokens,
            'response_tokens': response_tokens,
            'total_tokens': total,
            'context_ratio': context_tokens / total if total > 0 else 0
        }