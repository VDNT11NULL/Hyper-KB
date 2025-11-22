"""
Updated main script demonstrating the complete pipeline.
Shows curation, storage, and hybrid retrieval.
"""

from uuid import uuid4
from kb.kb_enhanced import EnhancedKnowledgeBase
from retrieval.sparse_retrieval import BM25Retriever, MongoFTSRetriever
from retrieval.dense_retrieval import FAISSRetriever
from retrieval.hybrid_fusion import HybridRetriever
from .orchestrator import CuratorLLM, RetrievalOrchestrator


def main():
    """Main execution function."""
    
    print("=" * 70)
    print("HYBRID RETRIEVAL SYSTEM - COMPLETE PIPELINE DEMO")
    print("=" * 70)
    
    # Initialize knowledge base
    print("\n[1/6] Initializing Knowledge Base...")
    kb = EnhancedKnowledgeBase(db_name="hyper_kb_demo")
    
    # Initialize curator
    print("\n[2/6] Initializing Curator...")
    curator_llm = CuratorLLM(kb)
    
    # Create sample session
    session_id = str(uuid4())
    print(f"\n[3/6] Creating session: {session_id}")
    
    # Sample interactions
    interactions = [
        {
            "query": """Can you explain the difference between transformer architecture and traditional 
            RNN/LSTM models? Also, why have transformers become so dominant in NLP tasks?""",
            "response": """Great question! The key differences between transformers and RNN/LSTM models are fundamental 
            to understanding modern NLP.

            Traditional RNNs and LSTMs process sequences sequentially - they read text word by word, 
            maintaining a hidden state that gets updated at each step. This sequential nature creates 
            two major problems: first, they struggle with long-range dependencies because information 
            from early in the sequence can get lost or diluted as it propagates through many time steps. 
            Second, this sequential processing prevents parallelization during training, making them 
            computationally inefficient.

            Transformers, introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al., 
            revolutionized this approach through the self-attention mechanism. Instead of processing 
            sequentially, transformers can attend to all positions in the input simultaneously. The 
            self-attention mechanism computes attention scores between every pair of tokens, allowing 
            the model to directly capture relationships regardless of distance in the sequence.

            The architecture consists of encoder and decoder blocks. Each encoder block has two main 
            components: a multi-head self-attention layer and a feed-forward neural network. Multi-head 
            attention allows the model to focus on different aspects of the input simultaneously - some 
            heads might focus on syntax, others on semantics, and others on long-range dependencies.

            Transformers became dominant for several reasons. First, they're highly parallelizable since 
            all tokens are processed simultaneously, enabling efficient training on GPUs. Second, they 
            handle long-range dependencies much better through direct attention connections. Third, they 
            scale remarkably well - larger transformers with more parameters consistently perform better, 
            leading to models like GPT-3, GPT-4, and Claude. Fourth, positional encodings allow them to 
            maintain sequence order information without sequential processing.

            The attention mechanism's interpretability is also valuable - we can visualize what the model 
            focuses on, providing insights into its decision-making process. This has led to transformers 
            dominating not just NLP, but also computer vision, protein folding, and multimodal tasks.""",
            "dialogue_act": "question"
        },
        {
            "query": "What are the computational requirements for training large transformer models?",
            "response": """Training large transformer models requires substantial computational resources. 
            Modern large language models like GPT-3 or GPT-4 require thousands of GPUs running for weeks or months. 
            The memory requirements scale with model size, sequence length, and batch size. For example, training 
            a model with billions of parameters might require hundreds of gigabytes of GPU memory distributed across 
            multiple devices. The training process also demands significant energy, with estimates suggesting that 
            training large models can produce carbon emissions equivalent to several transatlantic flights.""",
            "dialogue_act": "question",
            "topic_shift_score": 0.3
        }
    ]
    
    # Curate and store interactions
    print("\n[4/6] Curating and storing interactions...")
    interaction_ids = curator_llm.curate_batch(interactions, session_id)
    print(f"Stored {len(interaction_ids)} interactions")
    
    # Initialize retrievers
    print("\n[5/6] Initializing retrievers...")
    
    # Sparse retriever (using BM25)
    sparse_retriever = BM25Retriever(kb, k1=1.5, b=0.75)
    sparse_retriever.index_documents()
    
    # Dense retriever (using FAISS)
    dense_retriever = FAISSRetriever(kb, dimension=384, index_type='flat')
    dense_retriever.index_embeddings()
    
    # Hybrid retriever with RRF fusion
    hybrid_retriever = HybridRetriever(
        sparse_retriever=sparse_retriever,
        dense_retriever=dense_retriever,
        fusion_method='rrf',
        fusion_params={'k': 60}
    )
    
    # Initialize retrieval orchestrator
    retrieval_orch = RetrievalOrchestrator(kb, hybrid_retriever)
    
    # Perform retrieval
    print("\n[6/6] Testing retrieval...")
    test_query = "How do attention mechanisms work in transformers?"
    query_embedding = curator_llm.get_embedding_for_query(test_query)
    
    print(f"\nQuery: {test_query}")
    print("\n--- Sparse Retrieval Results ---")
    sparse_results = retrieval_orch.retrieve(
        query=test_query,
        query_embedding=query_embedding,
        session_id=session_id,
        top_k=3,
        method='sparse'
    )
    for result in sparse_results:
        print(f"Rank {result.rank}: Score={result.score:.4f}")
        print(f"  Query: {result.query_text[:80]}...")
        print()
    
    print("\n--- Dense Retrieval Results ---")
    dense_results = retrieval_orch.retrieve(
        query=test_query,
        query_embedding=query_embedding,
        session_id=session_id,
        top_k=3,
        method='dense'
    )
    for result in dense_results:
        print(f"Rank {result.rank}: Score={result.score:.4f}")
        print(f"  Query: {result.query_text[:80]}...")
        print()
    
    print("\n--- Hybrid Retrieval Results (RRF) ---")
    hybrid_results = retrieval_orch.retrieve(
        query=test_query,
        query_embedding=query_embedding,
        session_id=session_id,
        top_k=3,
        method='hybrid'
    )
    for result in hybrid_results:
        print(f"Rank {result.rank}: Score={result.score:.4f}")
        print(f"  Query: {result.query_text[:80]}...")
        print()
    
    # Show statistics
    print("\n" + "=" * 70)
    print("KNOWLEDGE BASE STATISTICS")
    print("=" * 70)
    stats = kb.get_stats()
    for key, value in stats.items():
        if key != 'most_accessed':
            print(f"{key}: {value}")
    
    # Close connection
    kb.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    main()