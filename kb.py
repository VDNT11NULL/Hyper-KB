# Pymongo install kar lena
from pymongo import MongoClient, ASCENDING, TEXT
from pymongo.server_api import ServerApi
from datetime import datetime
from typing import List, Dict, Optional
import uuid
from bson import Binary
import pickle
import numpy as np

class KnowledgeBase:
    """
    MongoDB-based Knowledge Base for storing and retrieving conversation interactions
    with support for hybrid retrieval (sparse + dense).
    """
    def __init__(self, db_name: str = "hyper_kb"):
        connection_string = "mongodb+srv://mayank:123mayank@cluster0.nfmeynh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        
        try:
            self.client = MongoClient(connection_string, server_api=ServerApi('1'))
            self.client.admin.command('ping')
            print("Successfully connected to MongoDB Atlas!")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            raise
        
        self.db = self.client[db_name]
        
        self.interactions = self.db['interactions']
        self.features = self.db['features']
        self.embeddings = self.db['embeddings']
        
        self._setup_indexes()
    
    def _setup_indexes(self):
        try:
            try:
                interaction_indexes = list(self.interactions.list_indexes())
                feature_indexes = list(self.features.list_indexes())
                
                for idx in interaction_indexes:
                    if idx.get('weights'): 
                        self.interactions.drop_index(idx['name'])
                        
                for idx in feature_indexes:
                    if idx.get('weights'):
                        self.features.drop_index(idx['name'])
            except:
                pass  
            
            # Interaction collection indexes
            self.interactions.create_index([('interaction_id', ASCENDING)], unique=True)
            self.interactions.create_index([('session_id', ASCENDING)])
            self.interactions.create_index([('timestamp', ASCENDING)])
            self.interactions.create_index([('query_text', TEXT), ('response_text', TEXT)])
            
            # Feature collection indexes
            self.features.create_index([('feature_id', ASCENDING)], unique=True)
            self.features.create_index([('interaction_id', ASCENDING)])
            self.features.create_index([('keywords', TEXT), ('context_passage', TEXT)])
            
            # Embedding collection indexes
            self.embeddings.create_index([('embedding_id', ASCENDING)], unique=True)
            self.embeddings.create_index([('interaction_id', ASCENDING)])
            
            print(" Indexes created successfully!")
        except Exception as e:
            print(f" Warning: Index creation issue: {e}")
    
    def insert_interaction(self, 
                          query_text: str, 
                          response_text: str, 
                          session_id: str,
                          keywords: List[str] = None,
                          entities: Dict = None,
                          context_passage: str = None,
                          embedding_vector: np.ndarray = None) -> str:
        """
        Insert a new interaction with all related features and embeddings.
        
        Args:
            query_text: User's query
            response_text: LLM's response
            session_id: Session identifier for conversation continuity
            keywords: List of extracted keywords
            entities: Dictionary of named entities
            context_passage: Contextual passage for the interaction
            embedding_vector: Dense vector representation
            
        Returns:
            interaction_id: UUID of the inserted interaction
        """
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        try:
            # Insert interaction
            interaction_doc = {
                'interaction_id': interaction_id,
                'query_text': query_text,
                'response_text': response_text,
                'timestamp': timestamp,
                'session_id': session_id
            }
            self.interactions.insert_one(interaction_doc)
            
            # Insert features if provided
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
            
            # Insert embedding if provided
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
            
            return interaction_id
        except Exception as e:
            print(f" Error inserting interaction: {e}")
            raise
   
    def get_interaction_by_id(self, interaction_id: str) -> Optional[Dict]:
        """Retrieve complete interaction with features."""
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
            
            return interaction
        except Exception as e:
            print(f" Error retrieving interaction: {e}")
            return None
    
    def get_session_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve conversation history for a session."""
        try:
            results = list(self.interactions.find(
                {'session_id': session_id}
            ).sort('timestamp', ASCENDING).limit(limit))
            
            for result in results:
                result.pop('_id', None)
            
            return results
        except Exception as e:
            print(f" Error retrieving session history: {e}")
            return []
    
    def update_interaction_features(self, 
                                   interaction_id: str,
                                   keywords: List[str] = None,
                                   entities: Dict = None,
                                   context_passage: str = None):
        """Update features for an existing interaction."""
        try:
            update_doc = {}
            if keywords is not None:
                update_doc['keywords'] = keywords
            if entities is not None:
                update_doc['entities'] = entities
            if context_passage is not None:
                update_doc['context_passage'] = context_passage
            
            if update_doc:
                result = self.features.update_one(
                    {'interaction_id': interaction_id},
                    {'$set': update_doc},
                    upsert=True
                )
                print(f" Updated features for interaction {interaction_id}")
                return result.modified_count > 0 or result.upserted_id is not None
            return False
        except Exception as e:
            print(f" Error updating features: {e}")
            return False
    
    def delete_interaction(self, interaction_id: str) -> bool:
        """Delete an interaction and all associated data."""
        try:
            # Delete from all collections
            self.interactions.delete_one({'interaction_id': interaction_id})
            self.features.delete_one({'interaction_id': interaction_id})
            self.embeddings.delete_one({'interaction_id': interaction_id})
            print(f" Deleted interaction {interaction_id}")
            return True
        except Exception as e:
            print(f" Error deleting interaction: {e}")
            return False
    
    def clear_all_data(self):
        """Clear all data from the knowledge base (use with caution!)."""
        try:
            self.interactions.delete_many({})
            self.features.delete_many({})
            self.embeddings.delete_many({})
            print(" All data cleared from knowledge base")
        except Exception as e:
            print(f" Error clearing data: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base."""
        try:
            stats = {
                'total_interactions': self.interactions.count_documents({}),
                'total_features': self.features.count_documents({}),
                'total_embeddings': self.embeddings.count_documents({}),
                'unique_sessions': len(self.interactions.distinct('session_id'))
            }
            return stats
        except Exception as e:
            print(f" Error getting stats: {e}")
            return {}
    
    def close(self):
        """Close MongoDB connection."""
        try:
            self.client.close()
            print(" MongoDB connection closed")
        except Exception as e:
            print(f" Error closing connection: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize Knowledge Base
    kb = KnowledgeBase()
    
    # Get current stats
    print("\n=== Knowledge Base Stats ===")
    stats = kb.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Example: Insert an interaction
    session_id = str(uuid.uuid4())
    print(f"\n=== Inserting New Interaction ===")
    print(f"Session ID: {session_id}")
    
    interaction_id = kb.insert_interaction(
        query_text="What is information retrieval?",
        response_text="Information retrieval is the process of obtaining relevant information from large repositories.",
        session_id=session_id,
        keywords=["information retrieval", "process", "repositories"],
        entities={"CONCEPT": ["information retrieval"]},
        context_passage="Information retrieval systems help users find relevant documents.",
        embedding_vector=np.random.rand(384)  # Randomly Generating the numbers
    )
    
    print(f" Inserted interaction: {interaction_id}")

    kb.close()