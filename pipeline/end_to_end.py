"""
Complete end-to-end pipeline integrating all components.
"""

from typing import Optional, Dict, List
from uuid import uuid4
import numpy as np
from datetime import datetime

from kb.kb_enhanced import EnhancedKnowledgeBase, InteractionMetadata
from retrieval.sparse_retrieval import BM25Retriever
from retrieval.dense_retrieval import FAISSRetriever
from retrieval.hybrid_fusion import HybridRetriever
from curator.orchestrator import CuratorLLM
from prompt_enhancement import ContextAggregator, PromptBuilder
from drift import DriftDetector, ContextTracker, AdaptiveRetriever


class HybridRetrievalPipeline:
    """
    Complete pipeline: Query -> Retrieval -> Context Enhancement -> Response Storage.
    Handles conversational drift automatically.
    """
    
    def __init__(
        self,
        db_name: str = "hyper_kb_production",
        fusion_method: str = 'rrf',
        aggregation_strategy: str = 'weighted',
        prompt_template: str = 'conversational'
    ):
        """Initialize complete pipeline."""
        print("Initializing Hybrid Retrieval Pipeline...")
        
        # Knowledge base
        self.kb = EnhancedKnowledgeBase(db_name=db_name)
        
        # Curator
        self.curator = CuratorLLM(self.kb)
        
        # Retrievers (initialize but don't index yet)
        self.sparse_retriever = None
        self.dense_retriever = None
        self.hybrid_retriever = None
        
        # Drift handling
        self.drift_detector = DriftDetector()
        self.context_tracker = ContextTracker()
        self.adaptive_retriever = None
        
        # Prompt enhancement
        self.context_aggregator = ContextAggregator(strategy=aggregation_strategy)
        self.prompt_builder = PromptBuilder(template_name=prompt_template)
        
        # State
        self.indexed = False
        self.fusion_method = fusion_method
        
        print("Pipeline initialized successfully!")
    
    def ensure_indexed(self):
        """Ensure all indexes are built."""
        if self.indexed:
            return
        
        print("Building indexes...")
        
        # Build sparse retriever
        self.sparse_retriever = BM25Retriever(self.kb)
        self.sparse_retriever.index_documents()
        
        # Build dense retriever
        self.dense_retriever = FAISSRetriever(self.kb, dimension=384)
        self.dense_retriever.index_embeddings()
        
        # Build hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            self.sparse_retriever,
            self.dense_retriever,
            fusion_method=self.fusion_method
        )
        
        # Build adaptive retriever
        self.adaptive_retriever = AdaptiveRetriever(
            self.hybrid_retriever,
            self.drift_detector,
            self.context_tracker
        )
        
        self.indexed = True
        print("Indexes built successfully!")
    
    def index_knowledge_base(self):
        """Index all documents for retrieval."""
        self.ensure_indexed()
    
    def process_interaction(
        self,
        query: str,
        response: str,
        session_id: str,
        store: bool = True
    ) -> Dict:
        """Process and store a complete interaction."""
        # Get session history
        history = self.kb.get_session_history(session_id, limit=10)
        
        # Detect drift
        topic_shift_score = 0.0
        shift_detected = False
        
        if history:
            prev = history[-1]
            shift_detected, similarity = self.drift_detector.detect_topic_shift(
                current_query=query,
                previous_query=prev['query_text'],
                current_response=response,
                previous_response=prev['response_text']
            )
            topic_shift_score = 1.0 - similarity
        
        # Store interaction if requested
        interaction_id = None
        if store:
            interaction_id = self.curator.curate_and_store(
                query_text=query,
                response_text=response,
                session_id=session_id,
                topic_shift_score=topic_shift_score
            )
            
            # Track in context tracker
            self.context_tracker.track_interaction(
                session_id=session_id,
                interaction_id=interaction_id,
                query=query,
                response=response,
                turn_number=len(history),
                topic_shift_detected=shift_detected,
                topic_shift_score=topic_shift_score
            )
            
            # Mark for re-indexing
            self.indexed = False
        
        return {
            'interaction_id': interaction_id,
            'shift_detected': shift_detected,
            'topic_shift_score': topic_shift_score,
            'session_turn': len(history)
        }
    
    def query(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        use_adaptive: bool = True
    ) -> Dict:
        """Complete query pipeline with context retrieval."""
        # Ensure indexes are built
        self.ensure_indexed()
        
        # Get query embedding
        query_embedding = self.curator.get_embedding_for_query(query)
        
        # Retrieve contexts
        if use_adaptive and self.adaptive_retriever:
            history = self.context_tracker.get_active_context_window(session_id)
            results = self.adaptive_retriever.retrieve_adaptive(
                query=query,
                query_embedding=query_embedding,
                session_id=session_id,
                conversation_history=history,
                top_k=top_k
            )
        else:
            results = self.hybrid_retriever.search(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k
            )
        
        # Get session history for context
        session_history = self.kb.get_session_history(session_id, limit=3)
        
        # Aggregate context
        aggregated_context = self.context_aggregator.aggregate(results)
        
        # Build enhanced prompt
        enhanced_prompt = self.prompt_builder.build(
            query=query,
            context=aggregated_context,
            metadata={
                'session_id': session_id,
                'turn_number': len(session_history),
                'topic_shift_detected': self.context_tracker.should_expand_search(session_id)
            }
        )
        
        return {
            'query': query,
            'enhanced_prompt': enhanced_prompt,
            'retrieved_contexts': len(results),
            'retrieval_results': results,
            'session_turn': len(session_history),
            'drift_state': self.context_tracker.get_session_summary(session_id)
        }
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics."""
        kb_stats = self.kb.get_stats()
        
        return {
            'kb_stats': kb_stats,
            'indexed': self.indexed,
            'active_sessions': len(self.context_tracker.session_states)
        }
    
    def close(self):
        """Cleanup resources."""
        self.kb.close()