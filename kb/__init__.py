"""
Knowledge Base package.
Provides storage backends and retrieval interfaces.
"""

from .kb_base import * 
from .kb_enhanced import EnhancedKnowledgeBase, KnowledgeBase

__all__ = [
    'StorageBackend',
    'SparseRetriever',
    'DenseRetriever',
    'InteractionMetadata',
    'RetrievalResult',
    'EnhancedKnowledgeBase',
    'KnowledgeBase'
]