"""
Context aggregation manager.
"""

from typing import List, Optional
from kb.kb_base import RetrievalResult
from .strategies import (
    AggregationStrategy,
    ConcatenationStrategy,
    TemplateStrategy,
    WeightedStrategy,
    RecencyAwareStrategy
)
from .context_utils import ContextUtils


class ContextAggregator:
    """
    Manages context aggregation with multiple strategies.
    """
    
    STRATEGIES = {
        'concatenate': ConcatenationStrategy,
        'template': TemplateStrategy,
        'weighted': WeightedStrategy,
        'recency': RecencyAwareStrategy
    }
    
    def __init__(
        self,
        strategy: str = 'concatenate',
        max_contexts: int = 5,
        max_tokens: int = 2000,
        deduplicate: bool = True,
        dedup_threshold: float = 0.85
    ):
        """
        Initialize aggregator.
        
        Args:
            strategy: Aggregation strategy name
            max_contexts: Maximum contexts to include
            max_tokens: Maximum total tokens
            deduplicate: Whether to remove duplicates
            dedup_threshold: Similarity threshold for deduplication
        """
        self.strategy_name = strategy
        self.strategy = self._create_strategy(strategy)
        self.max_contexts = max_contexts
        self.max_tokens = max_tokens
        self.deduplicate = deduplicate
        self.dedup_threshold = dedup_threshold
        self.utils = ContextUtils()
    
    def _create_strategy(self, strategy: str) -> AggregationStrategy:
        """Create strategy instance."""
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        return self.STRATEGIES[strategy]()
    
    def aggregate(
        self,
        results: List[RetrievalResult],
        session_history: Optional[List[RetrievalResult]] = None
    ) -> str:
        """
        Aggregate retrieval results into context string.
        
        Args:
            results: Retrieved results
            session_history: Optional session history to include
            
        Returns:
            Aggregated context string
        """
        if not results:
            return ""
        
        # Deduplicate if requested
        if self.deduplicate:
            results = self.utils.deduplicate_contexts(results, self.dedup_threshold)
        
        # Limit number of contexts
        results = self.utils.extract_most_relevant(results, self.max_contexts)
        
        # Add session history if provided
        if session_history:
            combined = list(session_history) + list(results)
            # Remove duplicates between history and results
            if self.deduplicate:
                combined = self.utils.deduplicate_contexts(combined, self.dedup_threshold)
            results = combined[:self.max_contexts]
        
        # Aggregate using strategy
        context = self.strategy.aggregate(results, self.max_tokens)
        
        return context