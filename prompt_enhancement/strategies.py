"""
Context aggregation strategies.
"""

from typing import List, Dict, Optional
from kb.kb_base import RetrievalResult
from .context_utils import ContextUtils


class AggregationStrategy:
    """Base class for aggregation strategies."""
    
    def aggregate(
        self,
        results: List[RetrievalResult],
        max_tokens: Optional[int] = None
    ) -> str:
        """Aggregate results into single context string."""
        raise NotImplementedError


class ConcatenationStrategy(AggregationStrategy):
    """Simple rank-ordered concatenation."""
    
    def __init__(self, separator: str = "\n\n---\n\n"):
        self.separator = separator
        self.utils = ContextUtils()
    
    def aggregate(
        self,
        results: List[RetrievalResult],
        max_tokens: Optional[int] = None
    ) -> str:
        """Concatenate contexts in rank order."""
        contexts = []
        total_tokens = 0
        
        for result in sorted(results, key=lambda r: r.rank):
            context = f"Q: {result.query_text}\nA: {result.response_text}"
            tokens = self.utils.estimate_tokens(context)
            
            if max_tokens and (total_tokens + tokens > max_tokens):
                break
            
            contexts.append(context)
            total_tokens += tokens
        
        return self.separator.join(contexts)


class TemplateStrategy(AggregationStrategy):
    """Template-based formatting."""
    
    def __init__(self):
        self.utils = ContextUtils()
    
    def aggregate(
        self,
        results: List[RetrievalResult],
        max_tokens: Optional[int] = None
    ) -> str:
        """Format as structured template."""
        contexts = ["# Retrieved Context\n"]
        total_tokens = 0
        
        for i, result in enumerate(sorted(results, key=lambda r: r.rank), 1):
            context = f"\n## Context {i} (Relevance: {result.score:.3f})\n"
            context += f"**Previous Query**: {result.query_text}\n"
            context += f"**Response**: {result.response_text}\n"
            
            tokens = self.utils.estimate_tokens(context)
            
            if max_tokens and (total_tokens + tokens > max_tokens):
                break
            
            contexts.append(context)
            total_tokens += tokens
        
        return "".join(contexts)


class WeightedStrategy(AggregationStrategy):
    """Relevance-weighted aggregation with emphasis markers."""
    
    def __init__(self):
        self.utils = ContextUtils()
    
    def aggregate(
        self,
        results: List[RetrievalResult],
        max_tokens: Optional[int] = None
    ) -> str:
        """Aggregate with relevance indicators."""
        if not results:
            return ""
        
        # Normalize scores
        scores = [r.score for r in results]
        max_score = max(scores) if scores else 1.0
        
        contexts = []
        total_tokens = 0
        
        for result in sorted(results, key=lambda r: r.rank):
            weight = result.score / max_score
            
            # Add emphasis based on weight
            if weight > 0.8:
                prefix = "[HIGHLY RELEVANT] "
            elif weight > 0.5:
                prefix = "[RELEVANT] "
            else:
                prefix = "[POTENTIALLY RELEVANT] "
            
            context = f"{prefix}Q: {result.query_text}\nA: {result.response_text}"
            tokens = self.utils.estimate_tokens(context)
            
            if max_tokens and (total_tokens + tokens > max_tokens):
                break
            
            contexts.append(context)
            total_tokens += tokens
        
        return "\n\n".join(contexts)


class RecencyAwareStrategy(AggregationStrategy):
    """Prioritize recent interactions for drift handling."""
    
    def __init__(self):
        self.utils = ContextUtils()
    
    def aggregate(
        self,
        results: List[RetrievalResult],
        max_tokens: Optional[int] = None
    ) -> str:
        """Aggregate with recency bias."""
        # Sort by timestamp (most recent first)
        sorted_results = sorted(
            results,
            key=lambda r: r.metadata.timestamp,
            reverse=True
        )
        
        contexts = []
        total_tokens = 0
        
        for result in sorted_results:
            turns_ago = result.metadata.turn_number
            context = f"[{turns_ago} turns ago] Q: {result.query_text}\nA: {result.response_text}"
            tokens = self.utils.estimate_tokens(context)
            
            if max_tokens and (total_tokens + tokens > max_tokens):
                break
            
            contexts.append(context)
            total_tokens += tokens
        
        return "\n\n".join(contexts)