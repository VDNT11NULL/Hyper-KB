"""
Migration script to convert data from original kb.py to enhanced schema.
"""

from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
from typing import Optional
import uuid


def migrate_kb_data(
    source_db: str = "hyper_kb",
    target_db: str = "hyper_kb_enhanced",
    connection_string: Optional[str] = None
):
    """
    Migrate data from original KB to enhanced KB schema.
    
    Args:
        source_db: Source database name
        target_db: Target database name
        connection_string: MongoDB connection string
    """
    if connection_string is None:
        connection_string = (
            "mongodb+srv://mayank:123mayank@cluster0.nfmeynh.mongodb.net/"
            "?retryWrites=true&w=majority&appName=Cluster0"
        )
    
    print("Connecting to MongoDB...")
    client = MongoClient(connection_string, server_api=ServerApi('1'))
    
    source = client[source_db]
    target = client[target_db]
    
    # Migrate interactions
    print(f"\nMigrating interactions from {source_db} to {target_db}...")
    interactions = list(source['interactions'].find({}))
    
    migrated_count = 0
    session_map = {}
    
    for interaction in interactions:
        session_id = interaction.get('session_id')
        
        # Track turn numbers per session
        if session_id not in session_map:
            session_map[session_id] = 0
        else:
            session_map[session_id] += 1
        
        turn_number = session_map[session_id]
        
        # Create enhanced metadata
        enhanced_interaction = {
            'interaction_id': interaction['interaction_id'],
            'query_text': interaction['query_text'],
            'response_text': interaction['response_text'],
            'session_id': session_id,
            'metadata': {
                'timestamp': interaction.get('timestamp', datetime.utcnow()),
                'turn_number': turn_number,
                'session_duration_so_far': 0.0,
                'previous_interaction_id': None,
                'next_interaction_id': None,
                'dialogue_act': None,
                'access_count': 0,
                'last_accessed': None,
                'retrieval_scores': [],
                'topic_shift_score': 0.0,
                'context_stability': 1.0,
                'user_rating': None,
                'was_useful': None
            }
        }
        
        # Insert into target
        try:
            target['interactions'].insert_one(enhanced_interaction)
            migrated_count += 1
        except Exception as e:
            print(f"Error migrating interaction {interaction['interaction_id']}: {e}")
    
    print(f"Migrated {migrated_count} interactions")
    
    # Migrate features
    print("\nMigrating features...")
    features = list(source['features'].find({}))
    for feature in features:
        feature.pop('_id', None)
        try:
            target['features'].insert_one(feature)
        except Exception as e:
            print(f"Error migrating feature: {e}")
    
    print(f"Migrated {len(features)} features")
    
    # Migrate embeddings
    print("\nMigrating embeddings...")
    embeddings = list(source['embeddings'].find({}))
    for embedding in embeddings:
        embedding.pop('_id', None)
        try:
            target['embeddings'].insert_one(embedding)
        except Exception as e:
            print(f"Error migrating embedding: {e}")
    
    print(f"Migrated {len(embeddings)} embeddings")
    
    # Create session documents
    print("\nCreating session documents...")
    for session_id, turn_count in session_map.items():
        session_interactions = list(target['interactions'].find(
            {'session_id': session_id}
        ).sort('metadata.timestamp', 1))
        
        if session_interactions:
            session_doc = {
                'session_id': session_id,
                'start_time': session_interactions[0]['metadata']['timestamp'],
                'last_updated': session_interactions[-1]['metadata']['timestamp'],
                'interaction_ids': [i['interaction_id'] for i in session_interactions],
                'turn_count': len(session_interactions)
            }
            
            try:
                target['sessions'].insert_one(session_doc)
            except Exception as e:
                print(f"Error creating session {session_id}: {e}")
    
    print(f"Created {len(session_map)} session documents")
    
    print("\nMigration complete!")
    client.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        source = sys.argv[1]
        target = sys.argv[2] if len(sys.argv) > 2 else "hyper_kb_enhanced"
    else:
        source = "hyper_kb"
        target = "hyper_kb_enhanced"
    
    print(f"Migrating from '{source}' to '{target}'")
    print("WARNING: This will create new collections in the target database.")
    response = input("Continue? (y/n): ")
    
    if response.lower() == 'y':
        migrate_kb_data(source, target)
    else:
        print("Migration cancelled")