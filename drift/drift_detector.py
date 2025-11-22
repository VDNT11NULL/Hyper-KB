"""
Topic shift and drift detection for conversational systems.
Core component for maintaining context coherence.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from datetime import datetime
from sentence_transformers import SentenceTransformer
from kb.kb_base import InteractionMetadata


class DriftDetector:
    """
    Detects topic shifts and context drift in conversations.
    
    Uses semantic similarity between consecutive turns to identify
    when conversation topic changes significantly.
    """
    
    def __init__(
        self,
        embedder=None,
        shift_threshold: float = 0.4,
        drift_window: int = 3
    ):
        """
        Initialize drift detector.
        
        Args:
            embedder: Sentence transformer for embeddings
            shift_threshold: Similarity threshold below which shift is detected
            drift_window: Number of previous turns to consider
        """
        self.embedder = embedder or SentenceTransformer("all-MiniLM-L6-v2")
        self.shift_threshold = shift_threshold
        self.drift_window = drift_window
    
    def detect_topic_shift(
        self,
        current_query: str,
        previous_query: str,
        current_response: Optional[str] = None,
        previous_response: Optional[str] = None
    ) -> Tuple[bool, float]:
        """
        Detect if topic shift occurred between consecutive turns.
        
        Args:
            current_query: Current user query
            previous_query: Previous user query
            current_response: Current response (optional)
            previous_response: Previous response (optional)
            
        Returns:
            Tuple of (shift_detected, similarity_score)
        """
        # Encode queries
        current_emb = self.embedder.encode(current_query)
        previous_emb = self.embedder.encode(previous_query)
        
        # Compute similarity
        query_sim = self._cosine_similarity(current_emb, previous_emb)
        
        # If responses provided, also compare those
        if current_response and previous_response:
            current_resp_emb = self.embedder.encode(current_response)
            previous_resp_emb = self.embedder.encode(previous_response)
            response_sim = self._cosine_similarity(current_resp_emb, previous_resp_emb)
            
            # Average query and response similarity
            similarity = (query_sim + response_sim) / 2.0
        else:
            similarity = query_sim
        
        shift_detected = similarity < self.shift_threshold
        
        return shift_detected, float(similarity)
    
    def compute_drift_score(
        self,
        current_query: str,
        conversation_history: list
    ) -> float:
        """
        Compute drift score over conversation window.
        
        Args:
            current_query: Current query
            conversation_history: List of previous interactions with 'query' field
            
        Returns:
            Drift score (0=stable, 1=high drift)
        """
        if not conversation_history:
            return 0.0
        
        # Look at last N turns
        recent_history = conversation_history[-self.drift_window:]
        
        current_emb = self.embedder.encode(current_query)
        
        similarities = []
        for interaction in recent_history:
            hist_query = interaction.get('query', interaction.get('query_text', ''))
            if hist_query:
                hist_emb = self.embedder.encode(hist_query)
                sim = self._cosine_similarity(current_emb, hist_emb)
                similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        # Drift is inverse of average similarity
        avg_similarity = np.mean(similarities)
        drift_score = 1.0 - avg_similarity
        
        return float(np.clip(drift_score, 0.0, 1.0))
    
    def detect_shift_from_metadata(
        self,
        current_metadata: InteractionMetadata,
        previous_metadata: Optional[InteractionMetadata] = None
    ) -> Dict[str, any]:
        """
        Detect shift using stored metadata.
        
        Args:
            current_metadata: Current interaction metadata
            previous_metadata: Previous interaction metadata
            
        Returns:
            Dict with shift detection results
        """
        results = {
            'shift_detected': False,
            'topic_shift_score': current_metadata.topic_shift_score,
            'context_stability': current_metadata.context_stability,
            'confidence': 0.0
        }
        
        if previous_metadata:
            # Compare stored topic shift scores
            if current_metadata.topic_shift_score > self.shift_threshold:
                results['shift_detected'] = True
            
            # Check context stability trend
            stability_delta = (
                current_metadata.context_stability - 
                previous_metadata.context_stability
            )
            
            results['stability_delta'] = stability_delta
            results['confidence'] = abs(current_metadata.topic_shift_score - 0.5) * 2
        
        return results
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
        
        return float(dot_product / norm_product)
    
    def adjust_threshold_adaptive(
        self,
        conversation_history: list,
        percentile: int = 25
    ):
        """
        Adaptively adjust threshold based on conversation characteristics.
        
        Args:
            conversation_history: Full conversation history
            percentile: Percentile for threshold (lower = more sensitive)
        """
        if len(conversation_history) < 2:
            return
        
        # Compute similarities between consecutive turns
        similarities = []
        for i in range(len(conversation_history) - 1):
            curr_query = conversation_history[i + 1].get('query', '')
            prev_query = conversation_history[i].get('query', '')
            
            if curr_query and prev_query:
                curr_emb = self.embedder.encode(curr_query)
                prev_emb = self.embedder.encode(prev_query)
                sim = self._cosine_similarity(curr_emb, prev_emb)
                similarities.append(sim)
        
        if similarities:
            # Set threshold at percentile of observed similarities
            self.shift_threshold = np.percentile(similarities, percentile)


class DriftAnalyzer:
    """
    Analyzes drift patterns across conversations.
    Provides insights for system optimization.
    """
    
    def __init__(self):
        self.detector = DriftDetector()
    
    def analyze_session_drift(
        self,
        session_interactions: list
    ) -> Dict[str, any]:
        """
        Analyze drift patterns in a complete session.
        
        Args:
            session_interactions: List of interactions in order
            
        Returns:
            Analysis results dict
        """
        if len(session_interactions) < 2:
            return {
                'num_interactions': len(session_interactions),
                'num_shifts': 0,
                'avg_drift_score': 0.0,
                'shift_points': []
            }
        
        shift_points = []
        drift_scores = []
        
        for i in range(1, len(session_interactions)):
            current = session_interactions[i]
            previous = session_interactions[i - 1]
            
            shift_detected, similarity = self.detector.detect_topic_shift(
                current_query=current.get('query_text', ''),
                previous_query=previous.get('query_text', ''),
                current_response=current.get('response_text', ''),
                previous_response=previous.get('response_text', '')
            )
            
            drift_score = 1.0 - similarity
            drift_scores.append(drift_score)
            
            if shift_detected:
                shift_points.append({
                    'turn': i,
                    'similarity': similarity,
                    'drift_score': drift_score
                })
        
        return {
            'num_interactions': len(session_interactions),
            'num_shifts': len(shift_points),
            'shift_rate': len(shift_points) / (len(session_interactions) - 1),
            'avg_drift_score': float(np.mean(drift_scores)) if drift_scores else 0.0,
            'max_drift_score': float(np.max(drift_scores)) if drift_scores else 0.0,
            'shift_points': shift_points
        }
    
    def compute_context_coherence(
        self,
        interactions: list,
        window_size: int = 3
    ) -> float:
        """
        Compute overall context coherence score for conversation.
        
        Args:
            interactions: List of interactions
            window_size: Window for computing coherence
            
        Returns:
            Coherence score (0-1, higher is better)
        """
        if len(interactions) < window_size:
            return 1.0
        
        coherence_scores = []
        
        for i in range(window_size - 1, len(interactions)):
            window = interactions[i - window_size + 1:i + 1]
            
            # Compute pairwise similarities in window
            queries = [w.get('query_text', '') for w in window]
            if not all(queries):
                continue
            
            embeddings = self.detector.embedder.encode(queries)
            
            # Average pairwise similarity
            sims = []
            for j in range(len(embeddings)):
                for k in range(j + 1, len(embeddings)):
                    sim = self.detector._cosine_similarity(
                        embeddings[j],
                        embeddings[k]
                    )
                    sims.append(sim)
            
            if sims:
                coherence_scores.append(np.mean(sims))
        
        return float(np.mean(coherence_scores)) if coherence_scores else 1.0