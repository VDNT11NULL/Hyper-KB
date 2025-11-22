"""
Context tracking and state management for conversations.
Maintains conversation state and topic transitions.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime
import numpy as np


class ConversationState:
    """Represents current conversation state."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.current_topic = None
        self.topic_history = []
        self.turn_count = 0
        self.last_shift_turn = 0
        self.active_contexts = []
        self.stability_score = 1.0
        
    def update(
        self,
        turn_number: int,
        topic: Optional[str] = None,
        shift_detected: bool = False
    ):
        """Update state with new turn information."""
        self.turn_count = turn_number
        
        if shift_detected:
            self.last_shift_turn = turn_number
            if topic and topic != self.current_topic:
                if self.current_topic:
                    self.topic_history.append(self.current_topic)
                self.current_topic = topic
        
        # Update stability (decay with shifts)
        turns_since_shift = turn_number - self.last_shift_turn
        self.stability_score = min(1.0, turns_since_shift / 5.0)
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary."""
        return {
            'session_id': self.session_id,
            'current_topic': self.current_topic,
            'topic_history': self.topic_history,
            'turn_count': self.turn_count,
            'last_shift_turn': self.last_shift_turn,
            'stability_score': self.stability_score
        }


class ContextTracker:
    """
    Tracks conversation context and manages state transitions.
    
    Maintains state for multiple active sessions and provides
    context-aware retrieval adjustments.
    """
    
    def __init__(
        self,
        max_history_length: int = 10,
        context_window: int = 5
    ):
        """
        Initialize context tracker.
        
        Args:
            max_history_length: Maximum history to maintain per session
            context_window: Sliding window size for active context
        """
        self.max_history_length = max_history_length
        self.context_window = context_window
        
        # Session management
        self.session_states = {}
        self.session_histories = defaultdict(lambda: deque(maxlen=max_history_length))
        self.topic_transitions = defaultdict(list)
    
    def track_interaction(
        self,
        session_id: str,
        interaction_id: str,
        query: str,
        response: str,
        turn_number: int,
        topic_shift_detected: bool = False,
        topic_shift_score: float = 0.0
    ):
        """
        Track new interaction in session.
        
        Args:
            session_id: Session identifier
            interaction_id: Interaction ID
            query: User query
            response: System response
            turn_number: Turn number in conversation
            topic_shift_detected: Whether shift was detected
            topic_shift_score: Magnitude of shift
        """
        # Initialize session state if needed
        if session_id not in self.session_states:
            self.session_states[session_id] = ConversationState(session_id)
        
        state = self.session_states[session_id]
        
        # Add to history
        interaction_data = {
            'interaction_id': interaction_id,
            'query': query,
            'response': response,
            'turn_number': turn_number,
            'timestamp': datetime.utcnow(),
            'topic_shift_detected': topic_shift_detected,
            'topic_shift_score': topic_shift_score
        }
        
        self.session_histories[session_id].append(interaction_data)
        
        # Update state
        state.update(turn_number, shift_detected=topic_shift_detected)
        
        # Track topic transition
        if topic_shift_detected:
            self.topic_transitions[session_id].append({
                'turn': turn_number,
                'shift_score': topic_shift_score
            })
    
    def get_active_context_window(
        self,
        session_id: str
    ) -> List[Dict]:
        """
        Get active context window for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of recent interactions in context window
        """
        history = self.session_histories.get(session_id, [])
        return list(history)[-self.context_window:]
    
    def get_retrieval_bias(
        self,
        session_id: str
    ) -> Dict[str, float]:
        """
        Compute retrieval bias parameters based on current state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict with bias parameters
        """
        state = self.session_states.get(session_id)
        
        if not state:
            return {
                'recency_weight': 0.5,
                'session_filter_strength': 0.5,
                'global_search_weight': 0.5
            }
        
        # High stability -> favor session context
        # Low stability (recent shift) -> favor global search
        stability = state.stability_score
        
        return {
            'recency_weight': 1.0 - stability * 0.5,  # Higher after shift
            'session_filter_strength': stability,      # Lower after shift
            'global_search_weight': 1.0 - stability   # Higher after shift
        }
    
    def should_expand_search(
        self,
        session_id: str,
        threshold: float = 0.3
    ) -> bool:
        """
        Determine if search should expand beyond session context.
        
        Args:
            session_id: Session identifier
            threshold: Stability threshold for expansion
            
        Returns:
            True if should expand search
        """
        state = self.session_states.get(session_id)
        if not state:
            return True
        
        # Expand if stability is low (recent shift)
        return state.stability_score < threshold
    
    def get_session_summary(
        self,
        session_id: str
    ) -> Dict:
        """
        Get comprehensive session summary.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary dict
        """
        state = self.session_states.get(session_id)
        history = self.session_histories.get(session_id, [])
        transitions = self.topic_transitions.get(session_id, [])
        
        return {
            'session_id': session_id,
            'state': state.to_dict() if state else None,
            'total_interactions': len(history),
            'topic_transitions': len(transitions),
            'active_context_size': len(self.get_active_context_window(session_id)),
            'retrieval_bias': self.get_retrieval_bias(session_id)
        }
    
    def reset_session(self, session_id: str):
        """Reset tracking for a session."""
        if session_id in self.session_states:
            del self.session_states[session_id]
        if session_id in self.session_histories:
            del self.session_histories[session_id]
        if session_id in self.topic_transitions:
            del self.topic_transitions[session_id]