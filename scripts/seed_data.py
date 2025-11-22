"""
Script to seed the knowledge base with sample conversational data.
Useful for testing and development.
"""

from uuid import uuid4
from datetime import datetime, timedelta
import random
from kb.kb_enhanced import EnhancedKnowledgeBase
from curator.orchestrator import CuratorLLM
from typing import Optional

SAMPLE_CONVERSATIONS = [
    {
        "session_name": "Machine Learning Basics",
        "interactions": [
            {
                "query": "What is machine learning?",
                "response": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
                "dialogue_act": "question"
            },
            {
                "query": "What are the main types of machine learning?",
                "response": "The three main types are: 1) Supervised learning, where models learn from labeled data; 2) Unsupervised learning, where models find patterns in unlabeled data; and 3) Reinforcement learning, where agents learn by interacting with an environment.",
                "dialogue_act": "question",
                "topic_shift_score": 0.2
            },
            {
                "query": "Can you explain supervised learning in more detail?",
                "response": "Supervised learning uses labeled training data to teach models to predict outputs. The model learns a mapping from inputs to outputs by comparing its predictions with the true labels and adjusting accordingly. Common examples include classification and regression tasks.",
                "dialogue_act": "question",
                "topic_shift_score": 0.1
            }
        ]
    },
    {
        "session_name": "Deep Learning Architecture",
        "interactions": [
            {
                "query": "What is a neural network?",
                "response": "A neural network is a computational model inspired by biological neural networks. It consists of layers of interconnected nodes (neurons) that process information. Each connection has a weight that adjusts as learning proceeds, allowing the network to learn complex patterns.",
                "dialogue_act": "question"
            },
            {
                "query": "What is backpropagation?",
                "response": "Backpropagation is the algorithm used to train neural networks. It calculates the gradient of the loss function with respect to each weight by propagating errors backward through the network, then updates weights to minimize the loss.",
                "dialogue_act": "question",
                "topic_shift_score": 0.3
            }
        ]
    },
    {
        "session_name": "Natural Language Processing",
        "interactions": [
            {
                "query": "What is tokenization in NLP?",
                "response": "Tokenization is the process of breaking text into smaller units called tokens, which can be words, subwords, or characters. It's a fundamental preprocessing step in NLP that converts raw text into a format that models can process.",
                "dialogue_act": "question"
            },
            {
                "query": "What are word embeddings?",
                "response": "Word embeddings are dense vector representations of words that capture semantic relationships. Words with similar meanings are mapped to nearby points in the embedding space. Popular methods include Word2Vec, GloVe, and more recently, contextualized embeddings from transformer models.",
                "dialogue_act": "question",
                "topic_shift_score": 0.2
            },
            {
                "query": "How do transformers differ from RNNs?",
                "response": "Transformers use self-attention mechanisms to process entire sequences in parallel, unlike RNNs which process sequentially. This allows transformers to better capture long-range dependencies and train much faster. They've become the dominant architecture in modern NLP.",
                "dialogue_act": "question",
                "topic_shift_score": 0.4
            }
        ]
    }
]


def seed_database(
    db_name: str = "hyper_kb_seed",
    num_sessions: Optional[int] = None
):
    """
    Seed database with sample conversations.
    
    Args:
        db_name: Database name to seed
        num_sessions: Number of sessions to create (None = use all samples)
    """
    print(f"Seeding database: {db_name}")
    print("=" * 70)
    
    # Initialize KB and curator
    kb = EnhancedKnowledgeBase(db_name=db_name)
    curator_llm = CuratorLLM(kb)
    
    # Clear existing data
    print("\nClearing existing data...")
    kb.interactions.delete_many({})
    kb.features.delete_many({})
    kb.embeddings.delete_many({})
    kb.sessions.delete_many({})
    kb.retrieval_logs.delete_many({})
    
    # Determine sessions to use
    sessions_to_seed = SAMPLE_CONVERSATIONS
    if num_sessions:
        sessions_to_seed = SAMPLE_CONVERSATIONS[:num_sessions]
    
    # Seed conversations
    total_interactions = 0
    for conv_idx, conversation in enumerate(sessions_to_seed, 1):
        session_id = str(uuid4())
        session_name = conversation['session_name']
        
        print(f"\n[{conv_idx}/{len(sessions_to_seed)}] Seeding session: {session_name}")
        print(f"Session ID: {session_id}")
        
        interactions = conversation['interactions']
        interaction_ids = curator_llm.curate_batch(interactions, session_id)
        
        print(f"  Stored {len(interaction_ids)} interactions")
        total_interactions += len(interaction_ids)
    
    # Show statistics
    print("\n" + "=" * 70)
    print("SEEDING COMPLETE")
    print("=" * 70)
    stats = kb.get_stats()
    print(f"Total sessions: {stats['unique_sessions']}")
    print(f"Total interactions: {stats['total_interactions']}")
    print(f"Total features: {stats['total_features']}")
    print(f"Total embeddings: {stats['total_embeddings']}")
    
    kb.close()


if __name__ == "__main__":
    import sys
    
    db_name = sys.argv[1] if len(sys.argv) > 1 else "hyper_kb_seed"
    seed_database(db_name)