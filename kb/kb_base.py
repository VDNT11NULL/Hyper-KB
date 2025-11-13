"""
Abstract base classes for pluggable Knowledge Base backends.
Supports multiple storage strategies while maintaining consistent interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass, field


@dataclass
class InteractionMetadata:
    """Rich metadata for conversational drift handling and retrieval optimization."""
    
    # Temporal metadata
    timestamp: datetime
    turn_number: int  # Position in conversation
    session_duration_so_far: float  # Seconds since session start
    
    # Conversational context
    session_id: str
    previous_interaction_id: Optional[str] = None
    next_interaction_id: Optional[str] = None
    dialogue_act: Optional[str] = None  # question, statement, command, etc.
    
    # Retrieval tracking (for drift analysis)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    retrieval_scores: List[float] = field(default_factory=list)  # Historical relevance
    
    # Drift indicators
    topic_shift_score: float = 0.0  # 0-1, how much topic changed from previous
    context_stability: float = 1.0  # 0-1, how stable the context is
    
    # User feedback (for evaluation)
    user_rating: Optional[int] = None  # 1-5 if provided
    was_useful: Optional[bool] = None


@dataclass
class RetrievalResult:
    """Container for retrieved interactions with metadata."""
    
    interaction_id: str
    query_text: str
    response_text: str
    score: float
    rank: int
    retrieval_method: str  # 'sparse', 'dense', 'hybrid'
    metadata: InteractionMetadata
    features: Optional[Dict] = None


class StorageBackend(ABC):
    """Abstract base class for KB storage backends."""
    
    @abstractmethod
    def insert_interaction(
        self,
        query_text: str,
        response_text: str,
        session_id: str,
        metadata: InteractionMetadata,
        keywords: List[str] = None,
        entities: Dict = None,
        context_passage: str = None,
        embedding_vector: np.ndarray = None
    ) -> str:
        """Insert a new interaction with all associated data."""
        pass
    
    @abstractmethod
    def get_interaction_by_id(self, interaction_id: str) -> Optional[Dict]:
        """Retrieve complete interaction by ID."""
        pass
    
    @abstractmethod
    def get_session_history(
        self, 
        session_id: str, 
        limit: int = 10,
        include_metadata: bool = True
    ) -> List[Dict]:
        """Get conversation history for a session."""
        pass
    
    @abstractmethod
    def update_retrieval_metadata(
        self,
        interaction_id: str,
        accessed_at: datetime,
        retrieval_score: float
    ):
        """Update metadata when interaction is retrieved."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base."""
        pass
    
    @abstractmethod
    def close(self):
        """Clean up resources."""
        pass


class SparseRetriever(ABC):
    """Abstract base class for sparse retrieval methods."""
    
    @abstractmethod
    def index_documents(self, documents: List[Dict]):
        """Index documents for sparse retrieval."""
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """Perform sparse retrieval."""
        pass


class DenseRetriever(ABC):
    """Abstract base class for dense retrieval methods."""
    
    @abstractmethod
    def index_embeddings(
        self,
        interaction_ids: List[str],
        embeddings: np.ndarray
    ):
        """Index embeddings for dense retrieval."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """Perform dense retrieval."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of stored embeddings."""
        pass