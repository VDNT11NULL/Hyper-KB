"""
Enhanced MongoDB Knowledge Base with hybrid retrieval support.
Extends original kb.py with rich metadata and retrieval capabilities.
"""

from pymongo import MongoClient, ASCENDING, TEXT
from pymongo.server_api import ServerApi
from datetime import datetime
from typing import List, Dict, Optional
import uuid
import numpy as np
from .kb_base import (
    StorageBackend, 
    InteractionMetadata, 
    RetrievalResult
)


class EnhancedKnowledgeBase(StorageBackend):
    """
    Enhanced MongoDB-based Knowledge Base with:
    - Rich metadata for drift handling
    - Session-aware conversation tracking
    - Retrieval optimization metadata
    - Backwards compatible with original kb.py
    """
    
    def __init__(
        self, 
        db_name: str = "hyper_kb_enhanced",
        connection_string: Optional[str] = None
    ):
        """
        Initialize enhanced knowledge base.
        
        Args:
            db_name: Database name
            connection_string: MongoDB connection string (uses default if None)
        """
        if connection_string is None:
            connection_string = (
                "mongodb+srv://mayank:123mayank@cluster0.nfmeynh.mongodb.net/"
                "?retryWrites=true&w=majority&appName=Cluster0"
            )
        
        try:
            self.client = MongoClient(connection_string, server_api=ServerApi('1'))
            self.client.admin.command('ping')
            print("✓ Successfully connected to MongoDB Atlas!")
        except Exception as e:
            print(f"✗ Failed to connect to MongoDB: {e}")
            raise
        
        self.db = self.client[db_name]
        
        # Collections
        self.interactions = self.db['interactions']
        self.features = self.db['features']
        self.embeddings = self.db['embeddings']
        self.sessions = self.db['sessions']  # New: session tracking
        self.retrieval_logs = self.db['retrieval_logs']  # New: retrieval analytics
        
        self._setup_indexes()
    
    def _setup_indexes(self):
        """Set up optimized indexes for hybrid retrieval."""
        try:
            # Clear old text indexes
            for collection in [self.interactions, self.features]:
                try:
                    for idx in list(collection.list_indexes()):
                        if idx.get('weights'):
                            collection.drop_index(idx['name'])
                except:
                    pass
            
            # Interaction collection indexes
            self.interactions.create_index([('interaction_id', ASCENDING)], unique=True)
            self.interactions.create_index([('session_id', ASCENDING)])
            self.interactions.create_index([('metadata.timestamp', ASCENDING)])
            self.interactions.create_index([('metadata.turn_number', ASCENDING)])
            self.interactions.create_index([
                ('query_text', TEXT), 
                ('response_text', TEXT)
            ])
            
            # Feature collection indexes
            self.features.create_index([('feature_id', ASCENDING)], unique=True)
            self.features.create_index([('interaction_id', ASCENDING)])
            self.features.create_index([
                ('keywords', TEXT), 
                ('context_passage', TEXT)
            ])
            
            # Embedding collection indexes
            self.embeddings.create_index([('embedding_id', ASCENDING)], unique=True)
            self.embeddings.create_index([('interaction_id', ASCENDING)])
            
            # Session collection indexes
            self.sessions.create_index([('session_id', ASCENDING)], unique=True)
            self.sessions.create_index([('start_time', ASCENDING)])
            
            # Retrieval logs indexes
            self.retrieval_logs.create_index([('timestamp', ASCENDING)])
            self.retrieval_logs.create_index([('session_id', ASCENDING)])
            
            print("✓ Indexes created successfully!")
        except Exception as e:
            print(f"⚠ Warning: Index creation issue: {e}")
    
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
        """
        Insert a new interaction with rich metadata.
        
        Args:
            query_text: User's query
            response_text: LLM's response
            session_id: Session identifier
            metadata: Rich metadata object
            keywords: Extracted keywords
            entities: Named entities
            context_passage: Contextual summary
            embedding_vector: Dense vector representation
            
        Returns:
            interaction_id: UUID of inserted interaction
        """
        interaction_id = str(uuid.uuid4())
        
        try:
            # Ensure session exists
            self._ensure_session_exists(session_id, metadata.timestamp)
            
            # Insert interaction with metadata
            interaction_doc = {
                'interaction_id': interaction_id,
                'query_text': query_text,
                'response_text': response_text,
                'session_id': session_id,
                'metadata': {
                    'timestamp': metadata.timestamp,
                    'turn_number': metadata.turn_number,
                    'session_duration_so_far': metadata.session_duration_so_far,
                    'previous_interaction_id': metadata.previous_interaction_id,
                    'next_interaction_id': metadata.next_interaction_id,
                    'dialogue_act': metadata.dialogue_act,
                    'access_count': metadata.access_count,
                    'last_accessed': metadata.last_accessed,
                    'retrieval_scores': metadata.retrieval_scores,
                    'topic_shift_score': metadata.topic_shift_score,
                    'context_stability': metadata.context_stability,
                    'user_rating': metadata.user_rating,
                    'was_useful': metadata.was_useful
                }
            }
            self.interactions.insert_one(interaction_doc)
            
            # Insert features
            if keywords or entities or context_passage:
                feature_id = str(uuid.uuid4())
                feature_doc = {
                    'feature_id': feature_id,
                    'interaction_id': interaction_id,
                    'keywords': keywords or [],
                    'entities': entities or {},
                    'context_passage': context_passage or ''
                }
                self.features.insert_one(feature_doc)
            
            # Insert embedding
            if embedding_vector is not None:
                embedding_id = str(uuid.uuid4())
                embedding_list = embedding_vector.tolist()
                embedding_doc = {
                    'embedding_id': embedding_id,
                    'interaction_id': interaction_id,
                    'embedding_vector': embedding_list,
                    'dimension': len(embedding_list)
                }
                self.embeddings.insert_one(embedding_doc)
            
            # Update session
            self._update_session(session_id, interaction_id, metadata)
            
            return interaction_id
            
        except Exception as e:
            print(f"✗ Error inserting interaction: {e}")
            raise
    
    def _ensure_session_exists(self, session_id: str, timestamp: datetime):
        """Create session document if it doesn't exist."""
        if not self.sessions.find_one({'session_id': session_id}):
            session_doc = {
                'session_id': session_id,
                'start_time': timestamp,
                'last_updated': timestamp,
                'interaction_ids': [],
                'turn_count': 0
            }
            self.sessions.insert_one(session_doc)
    
    def _update_session(
        self, 
        session_id: str, 
        interaction_id: str,
        metadata: InteractionMetadata
    ):
        """Update session with new interaction."""
        self.sessions.update_one(
            {'session_id': session_id},
            {
                '$push': {'interaction_ids': interaction_id},
                '$set': {'last_updated': metadata.timestamp},
                '$inc': {'turn_count': 1}
            }
        )
    
    def get_interaction_by_id(self, interaction_id: str) -> Optional[Dict]:
        """
        Retrieve complete interaction with features.
        
        Args:
            interaction_id: UUID of interaction
            
        Returns:
            Complete interaction document with features, or None
        """
        try:
            interaction = self.interactions.find_one({'interaction_id': interaction_id})
            if not interaction:
                return None
            
            interaction.pop('_id', None)
            
            # Get associated features
            features = self.features.find_one({'interaction_id': interaction_id})
            if features:
                features.pop('_id', None)
                interaction['features'] = features
            
            # Get embedding info (not the vector itself, just metadata)
            embedding = self.embeddings.find_one(
                {'interaction_id': interaction_id},
                {'embedding_id': 1, 'dimension': 1}
            )
            if embedding:
                embedding.pop('_id', None)
                interaction['embedding_info'] = embedding
            
            return interaction
            
        except Exception as e:
            print(f"✗ Error retrieving interaction: {e}")
            return None
    
    def get_session_history(
        self, 
        session_id: str, 
        limit: int = 10,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Retrieve conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of interactions to return
            include_metadata: Whether to include full metadata
            
        Returns:
            List of interaction documents ordered by timestamp
        """
        try:
            projection = None if include_metadata else {
                '_id': 0,
                'interaction_id': 1,
                'query_text': 1,
                'response_text': 1,
                'metadata.timestamp': 1,
                'metadata.turn_number': 1
            }
            
            results = list(self.interactions.find(
                {'session_id': session_id},
                projection
            ).sort('metadata.timestamp', ASCENDING).limit(limit))
            
            for result in results:
                result.pop('_id', None)
            
            return results
            
        except Exception as e:
            print(f"✗ Error retrieving session history: {e}")
            return []
    
    def update_retrieval_metadata(
        self,
        interaction_id: str,
        accessed_at: datetime,
        retrieval_score: float
    ):
        """
        Update metadata when interaction is retrieved.
        
        Args:
            interaction_id: UUID of retrieved interaction
            accessed_at: Timestamp of retrieval
            retrieval_score: Relevance score from retrieval
        """
        try:
            self.interactions.update_one(
                {'interaction_id': interaction_id},
                {
                    '$inc': {'metadata.access_count': 1},
                    '$set': {'metadata.last_accessed': accessed_at},
                    '$push': {'metadata.retrieval_scores': retrieval_score}
                }
            )
        except Exception as e:
            print(f"✗ Error updating retrieval metadata: {e}")
    
    def log_retrieval(
        self,
        session_id: str,
        query: str,
        retrieved_ids: List[str],
        scores: List[float],
        method: str
    ):
        """
        Log retrieval event for analysis.
        
        Args:
            session_id: Current session
            query: Query text
            retrieved_ids: List of retrieved interaction IDs
            scores: Corresponding scores
            method: Retrieval method used
        """
        try:
            log_doc = {
                'log_id': str(uuid.uuid4()),
                'timestamp': datetime.utcnow(),
                'session_id': session_id,
                'query': query,
                'retrieved_ids': retrieved_ids,
                'scores': scores,
                'method': method,
                'top_k': len(retrieved_ids)
            }
            self.retrieval_logs.insert_one(log_doc)
        except Exception as e:
            print(f"⚠ Warning: Failed to log retrieval: {e}")
    
    def get_stats(self) -> Dict:
        """
        Get comprehensive statistics about the knowledge base.
        
        Returns:
            Dictionary with various statistics
        """
        try:
            stats = {
                'total_interactions': self.interactions.count_documents({}),
                'total_features': self.features.count_documents({}),
                'total_embeddings': self.embeddings.count_documents({}),
                'unique_sessions': len(self.interactions.distinct('session_id')),
                'total_sessions': self.sessions.count_documents({}),
                'total_retrievals': self.retrieval_logs.count_documents({})
            }
            
            # Average turns per session
            pipeline = [
                {'$group': {
                    '_id': None,
                    'avg_turns': {'$avg': '$turn_count'}
                }}
            ]
            result = list(self.sessions.aggregate(pipeline))
            if result:
                stats['avg_turns_per_session'] = round(result[0]['avg_turns'], 2)
            
            # Most accessed interactions
            pipeline = [
                {'$sort': {'metadata.access_count': -1}},
                {'$limit': 5},
                {'$project': {
                    'interaction_id': 1,
                    'access_count': '$metadata.access_count'
                }}
            ]
            stats['most_accessed'] = list(self.interactions.aggregate(pipeline))
            
            return stats
            
        except Exception as e:
            print(f"✗ Error getting stats: {e}")
            return {}
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get detailed session information."""
        try:
            session = self.sessions.find_one({'session_id': session_id})
            if session:
                session.pop('_id', None)
            return session
        except Exception as e:
            print(f"✗ Error getting session info: {e}")
            return None
    
    def close(self):
        """Close MongoDB connection."""
        try:
            self.client.close()
            print("✓ MongoDB connection closed")
        except Exception as e:
            print(f"✗ Error closing connection: {e}")


# Backwards compatibility wrapper
class KnowledgeBase(EnhancedKnowledgeBase):
    """
    Backwards-compatible wrapper for original KnowledgeBase class.
    Automatically converts simple calls to enhanced format.
    """
    
    def insert_interaction(
        self,
        query_text: str,
        response_text: str,
        session_id: str,
        keywords: List[str] = None,
        entities: Dict = None,
        context_passage: str = None,
        embedding_vector: np.ndarray = None,
        metadata: Optional[InteractionMetadata] = None,
        turn_number: int = 0
    ) -> str:
        """
        Backwards-compatible insert method.
        Automatically creates metadata if not provided.
        """
        if metadata is None:
            # Auto-generate metadata
            session_info = self.get_session_info(session_id)
            if session_info:
                turn_number = session_info['turn_count']
                start_time = session_info['start_time']
                duration = (datetime.utcnow() - start_time).total_seconds()
            else:
                duration = 0.0
            
            metadata = InteractionMetadata(
                timestamp=datetime.utcnow(),
                turn_number=turn_number,
                session_duration_so_far=duration,
                session_id=session_id
            )
        
        return super().insert_interaction(
            query_text=query_text,
            response_text=response_text,
            session_id=session_id,
            metadata=metadata,
            keywords=keywords,
            entities=entities,
            context_passage=context_passage,
            embedding_vector=embedding_vector
        )