"""
Comprehensive test for drift handling system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from uuid import uuid4
from kb.kb_enhanced import EnhancedKnowledgeBase
from retrieval.sparse_retrieval import BM25Retriever
from retrieval.dense_retrieval import FAISSRetriever
from retrieval.hybrid_fusion import HybridRetriever
from curator.orchestrator import CuratorLLM
from drift import DriftDetector, ContextTracker, AdaptiveRetriever, DriftAnalyzer


def main():
    print("=" * 70)
    print("DRIFT HANDLING SYSTEM TEST")
    print("=" * 70)
    
    # Initialize system
    kb = EnhancedKnowledgeBase(db_name="hyper_kb_drift_test")
    curator = CuratorLLM(kb)
    
    # Setup retrievers
    sparse = BM25Retriever(kb)
    dense = FAISSRetriever(kb, dimension=384)
    hybrid = HybridRetriever(sparse, dense, fusion_method='rrf')
    
    # Initialize drift components
    drift_detector = DriftDetector(shift_threshold=0.4)
    context_tracker = ContextTracker()
    adaptive_retriever = AdaptiveRetriever(hybrid, drift_detector, context_tracker)
    
    # Simulate conversation with topic shift
    session_id = str(uuid4())
    
    conversations = [
        # Topic 1: Machine Learning
        {
            "query": "What is supervised learning?",
            "response": "Supervised learning uses labeled data to train models."
        },
        {
            "query": "How does gradient descent work?",
            "response": "Gradient descent iteratively minimizes loss by updating parameters."
        },
        # TOPIC SHIFT
        {
            "query": "Tell me about cooking pasta.",
            "response": "Pasta should be cooked in boiling salted water until al dente."
        },
        {
            "query": "What's the best type of pasta for carbonara?",
            "response": "Spaghetti or rigatoni work well for carbonara."
        }
    ]
    
    print("\n[1] Storing interactions and detecting drift...")
    
    prev_query = None
    prev_response = None
    interaction_ids = []
    
    for i, conv in enumerate(conversations):
        query = conv['query']
        response = conv['response']
        
        # Detect shift
        if prev_query:
            shift_detected, similarity = drift_detector.detect_topic_shift(
                current_query=query,
                previous_query=prev_query,
                current_response=response,
                previous_response=prev_response
            )
            
            topic_shift_score = 1.0 - similarity
            
            print(f"\nTurn {i}:")
            print(f"  Query: {query[:50]}...")
            print(f"  Shift detected: {shift_detected}")
            print(f"  Similarity: {similarity:.3f}")
            print(f"  Drift score: {topic_shift_score:.3f}")
        else:
            shift_detected = False
            topic_shift_score = 0.0
        
        # Store with shift info
        interaction_id = curator.curate_and_store(
            query_text=query,
            response_text=response,
            session_id=session_id,
            topic_shift_score=topic_shift_score
        )
        
        interaction_ids.append(interaction_id)
        
        # Track in context tracker
        context_tracker.track_interaction(
            session_id=session_id,
            interaction_id=interaction_id,
            query=query,
            response=response,
            turn_number=i,
            topic_shift_detected=shift_detected,
            topic_shift_score=topic_shift_score
        )
        
        prev_query = query
        prev_response = response
    
    # Index for retrieval
    sparse.index_documents()
    dense.index_embeddings()
    
    print("\n[2] Testing adaptive retrieval...")
    
    # Query in new topic context
    test_query = "How should I cook spaghetti?"
    query_emb = curator.get_embedding_for_query(test_query)
    
    history = context_tracker.get_active_context_window(session_id)
    
    results = adaptive_retriever.retrieve_adaptive(
        query=test_query,
        query_embedding=query_emb,
        session_id=session_id,
        conversation_history=history,
        top_k=3
    )
    
    print(f"\nAdaptive retrieval for: '{test_query}'")
    print(f"Retrieved {len(results)} results:")
    for r in results:
        print(f"  Rank {r.rank}: {r.query_text[:50]}... (score: {r.score:.3f})")
    
    # Analyze session drift
    print("\n[3] Analyzing session drift patterns...")
    
    analyzer = DriftAnalyzer()
    session_interactions = [
        {'query_text': conv['query'], 'response_text': conv['response']}
        for conv in conversations
    ]
    
    analysis = analyzer.analyze_session_drift(session_interactions)
    
    print(f"\nDrift Analysis:")
    print(f"  Total interactions: {analysis['num_interactions']}")
    print(f"  Topic shifts detected: {analysis['num_shifts']}")
    print(f"  Shift rate: {analysis['shift_rate']:.2%}")
    print(f"  Average drift score: {analysis['avg_drift_score']:.3f}")
    print(f"  Max drift score: {analysis['max_drift_score']:.3f}")
    
    if analysis['shift_points']:
        print(f"\n  Shift points:")
        for sp in analysis['shift_points']:
            print(f"    Turn {sp['turn']}: drift={sp['drift_score']:.3f}")
    
    # Context coherence
    coherence = analyzer.compute_context_coherence(session_interactions)
    print(f"\n  Context coherence: {coherence:.3f}")
    
    # Session summary
    print("\n[4] Session summary...")
    summary = context_tracker.get_session_summary(session_id)
    
    print(f"\nSession: {summary['session_id']}")
    print(f"  Total interactions: {summary['total_interactions']}")
    print(f"  Topic transitions: {summary['topic_transitions']}")
    print(f"  Active context size: {summary['active_context_size']}")
    print(f"\n  Retrieval bias:")
    for key, value in summary['retrieval_bias'].items():
        print(f"    {key}: {value:.3f}")
    
    kb.close()
    print("\n" + "=" * 70)
    print("DRIFT HANDLING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()