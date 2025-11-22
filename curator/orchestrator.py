"""
Updated orchestrator integrating enhanced KB and hybrid retrieval.
Manages the complete pipeline from curation to storage and retrieval.
"""

import numpy as np
from typing import List, Optional, Dict
from datetime import datetime
from .curator_module import CuratorModule
from kb.kb_enhanced import EnhancedKnowledgeBase
from kb.kb_base import InteractionMetadata


class CuratorLLM:
    """
    Orchestrator for LLM-based curation and knowledge base management.
    
    Handles:
    - Interaction curation (keywords, entities, context)
    - Embedding generation
    - Storage in enhanced knowledge base
    - Metadata management
    """
    
    def __init__(
        self,
        kb: EnhancedKnowledgeBase,
        embed_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize curator orchestrator.
        
        Args:
            kb: Enhanced knowledge base instance
            embed_model: Sentence transformer model name
        """
        self.kb = kb
        self.curator = CuratorModule(embed_model=embed_model)
        self.session_turn_counters = {}
        self.embedder = self.curator.embedder
    
    def curate_and_store(
        self,
        query_text: str,
        response_text: str,
        session_id: str,
        previous_interaction_id: Optional[str] = None,
        dialogue_act: Optional[str] = None,
        topic_shift_score: float = 0.0
    ) -> str:
        """
        Curate interaction and store in knowledge base.
        
        Args:
            query_text: User's query
            response_text: LLM's response
            session_id: Session identifier
            previous_interaction_id: ID of previous interaction in session
            dialogue_act: Type of dialogue act (question, statement, etc.)
            topic_shift_score: Score indicating topic shift from previous turn
            
        Returns:
            interaction_id: UUID of stored interaction
        """
        # Get turn number for this session
        turn_number = self.session_turn_counters.get(session_id, 0)
        self.session_turn_counters[session_id] = turn_number + 1
        
        # Calculate session duration
        session_info = self.kb.get_session_info(session_id)
        if session_info:
            start_time = session_info['start_time']
            duration = (datetime.utcnow() - start_time).total_seconds()
        else:
            duration = 0.0
        
        # Curate the interaction
        output = self.curator(query_text=query_text, response_text=response_text)
        
        # Generate embedding
        text_to_embed = output.get("context_passage") or (query_text + " " + response_text)
        embedding_vector = self.curator.embedder.encode(text_to_embed)
        
        # Create metadata
        metadata = InteractionMetadata(
            timestamp=datetime.utcnow(),
            turn_number=turn_number,
            session_duration_so_far=duration,
            session_id=session_id,
            previous_interaction_id=previous_interaction_id,
            dialogue_act=dialogue_act,
            topic_shift_score=topic_shift_score,
            context_stability=1.0 - topic_shift_score  # Inverse relationship
        )
        
        # Store in knowledge base
        interaction_id = self.kb.insert_interaction(
            query_text=query_text,
            response_text=response_text,
            session_id=session_id,
            metadata=metadata,
            keywords=output.get("keywords", []),
            entities=output.get("entities", {}),
            context_passage=output.get("context_passage", ""),
            embedding_vector=np.array(embedding_vector, dtype=float)
        )
        
        return interaction_id
    
    def curate_batch(
        self,
        interactions: List[Dict],
        session_id: str
    ) -> List[str]:
        """
        Curate and store multiple interactions in batch.
        
        Args:
            interactions: List of interaction dictionaries with 'query' and 'response'
            session_id: Session identifier
            
        Returns:
            List of interaction IDs
        """
        interaction_ids = []
        previous_id = None
        
        for idx, interaction in enumerate(interactions):
            query = interaction.get('query', '')
            response = interaction.get('response', '')
            dialogue_act = interaction.get('dialogue_act')
            topic_shift = interaction.get('topic_shift_score', 0.0)
            
            interaction_id = self.curate_and_store(
                query_text=query,
                response_text=response,
                session_id=session_id,
                previous_interaction_id=previous_id,
                dialogue_act=dialogue_act,
                topic_shift_score=topic_shift
            )
            
            interaction_ids.append(interaction_id)
            previous_id = interaction_id
        
        return interaction_ids
    
    def get_embedding_for_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        return np.array(self.curator.embedder.encode(query), dtype=float)
    
    def reset_session_counter(self, session_id: str):
        """Reset turn counter for a session."""
        self.session_turn_counters[session_id] = 0


class RetrievalOrchestrator:
    """
    Orchestrator for hybrid retrieval operations.
    
    Manages retrieval across different methods and handles
    result logging and metadata updates.
    """
    
    def __init__(
        self,
        kb: EnhancedKnowledgeBase,
        hybrid_retriever
    ):
        """
        Initialize retrieval orchestrator.
        
        Args:
            kb: Enhanced knowledge base instance
            hybrid_retriever: HybridRetriever instance
        """
        self.kb = kb
        self.hybrid_retriever = hybrid_retriever
    
    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        session_id: str,
        top_k: int = 10,
        method: str = 'hybrid',
        filters: Optional[Dict] = None
    ) -> List:
        """
        Perform retrieval and update metadata.
        
        Args:
            query: Query text
            query_embedding: Query embedding
            session_id: Current session ID
            top_k: Number of results to return
            method: 'sparse', 'dense', or 'hybrid'
            filters: Optional filters
            
        Returns:
            List of RetrievalResult objects
        """
        # Perform retrieval based on method
        if method == 'sparse':
            results = self.hybrid_retriever.search_sparse_only(
                query=query,
                top_k=top_k,
                filters=filters
            )
        elif method == 'dense':
            results = self.hybrid_retriever.search_dense_only(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters
            )
        elif method == 'hybrid':
            results = self.hybrid_retriever.search(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters
            )
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        
        # Update retrieval metadata
        timestamp = datetime.utcnow()
        for result in results:
            self.kb.update_retrieval_metadata(
                interaction_id=result.interaction_id,
                accessed_at=timestamp,
                retrieval_score=result.score
            )
        
        # Log retrieval
        retrieved_ids = [r.interaction_id for r in results]
        scores = [r.score for r in results]
        self.kb.log_retrieval(
            session_id=session_id,
            query=query,
            retrieved_ids=retrieved_ids,
            scores=scores,
            method=method
        )
        
        return results