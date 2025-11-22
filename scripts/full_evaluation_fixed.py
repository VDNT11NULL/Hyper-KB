"""
Fixed evaluation with proper data seeding and plot generation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from uuid import uuid4
from pipeline.end_to_end import HybridRetrievalPipeline
from retrieval.sparse_retrieval import BM25Retriever
from retrieval.dense_retrieval import FAISSRetriever
from retrieval.hybrid_fusion import HybridRetriever
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def seed_rich_data(pipeline):
    """Seed with diverse data."""
    conversations = [
        # AI/ML Topic
        ("What is machine learning?", "ML is AI that learns from data without explicit programming."),
        ("Explain neural networks", "Neural networks are computing systems inspired by biological brains with interconnected nodes."),
        ("What are transformers?", "Transformers are neural architectures using self-attention, introduced in 2017 for NLP."),
        
        # Programming Topic
        ("What is Python?", "Python is a high-level interpreted programming language known for simplicity."),
        ("Explain functions in Python", "Functions are reusable blocks of code defined with def keyword."),
        
        # Cooking Topic  
        ("How to cook pasta?", "Boil salted water, add pasta, cook 8-10 minutes until al dente, drain."),
        ("What is carbonara?", "Carbonara is Italian pasta dish with eggs, cheese, pancetta, and black pepper."),
        
        # Science Topic
        ("What is photosynthesis?", "Photosynthesis is the process where plants convert light into chemical energy."),
        ("Explain gravity", "Gravity is the force of attraction between objects with mass."),
        
        # History Topic
        ("When did WWII start?", "World War II started in 1939 when Germany invaded Poland."),
    ]
    
    print("Seeding database with 10 interactions...")
    for q, r in conversations:
        pipeline.process_interaction(q, r, str(uuid4()))
    print("Seeding complete!")

def run_evaluation():
    """Run complete evaluation with plots."""
    
    print("="*70)
    print("COMPLETE EVALUATION WITH PLOTS")
    print("="*70)
    
    # Initialize
    pipeline = HybridRetrievalPipeline(db_name="eval_final")
    
    # Seed data
    seed_rich_data(pipeline)
    pipeline.ensure_indexed()
    
    # Test queries
    test_queries = [
        "What is AI?",
        "Explain transformers in NLP",
        "How do I cook Italian food?",
        "Tell me about Python programming"
    ]
    
    methods = ['bm25', 'faiss', 'hybrid']
    results = {m: {'scores': [], 'num_results': []} for m in methods}
    
    print("\n" + "="*70)
    print("RUNNING RETRIEVAL TESTS")
    print("="*70)
    
    # Rebuild retrievers for each method
    sparse = BM25Retriever(pipeline.kb)
    sparse.index_documents()
    
    dense = FAISSRetriever(pipeline.kb, dimension=384)
    dense.index_embeddings()
    
    hybrid = HybridRetriever(sparse, dense, fusion_method='rrf')
    
    retrievers = {
        'bm25': sparse,
        'faiss': dense,
        'hybrid': hybrid
    }
    
    # Run tests
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        
        query_emb = pipeline.curator.embedder.encode(query)
        
        for method in methods:
            if method == 'bm25':
                res = retrievers['bm25'].search(query, top_k=5)
            elif method == 'faiss':
                res = retrievers['faiss'].search(query_emb, top_k=5)
            else:
                res = retrievers['hybrid'].search(query, query_emb, top_k=5)
            
            num_res = len(res)
            avg_score = np.mean([r.score for r in res]) if res else 0.0
            
            results[method]['scores'].append(avg_score)
            results[method]['num_results'].append(num_res)
            
            print(f"  {method.upper()}: {num_res} results, avg_score={avg_score:.3f}")
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    output_dir = Path("experiments/results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Score comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(test_queries))
    width = 0.25
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, method in enumerate(methods):
        ax1.bar(x + i*width, results[method]['scores'], width, 
                label=method.upper(), color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Query Index')
    ax1.set_ylabel('Average Score')
    ax1.set_title('Retrieval Score by Method')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f'Q{i+1}' for i in range(len(test_queries))])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Number of results
    for i, method in enumerate(methods):
        ax2.bar(x + i*width, results[method]['num_results'], width,
                label=method.upper(), color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Query Index')
    ax2.set_ylabel('Retrieved Contexts')
    ax2.set_title('Number of Retrieved Contexts')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'Q{i+1}' for i in range(len(test_queries))])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot1 = output_dir / "method_comparison.png"
    plt.savefig(plot1, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot1}")
    plt.close()
    
    # Plot 3: Average performance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_scores = [np.mean(results[m]['scores']) for m in methods]
    std_scores = [np.std(results[m]['scores']) for m in methods]
    
    bars = ax.bar(methods, avg_scores, yerr=std_scores, capsize=10,
                  color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Average Score')
    ax.set_xlabel('Method')
    ax.set_title('Average Retrieval Performance')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plot2 = output_dir / "average_performance.png"
    plt.savefig(plot2, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot2}")
    plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    for method in methods:
        avg = np.mean(results[method]['scores'])
        std = np.std(results[method]['scores'])
        print(f"\n{method.upper()}:")
        print(f"  Avg Score: {avg:.4f} ± {std:.4f}")
        print(f"  Avg Results: {np.mean(results[method]['num_results']):.1f}")
    
    # Best method
    best_method = max(methods, key=lambda m: np.mean(results[m]['scores']))
    print(f"\nBest Method: {best_method.upper()}")
    
    pipeline.close()
    
    print("\n" + "="*70)
    print("COMPLETE! Check experiments/results/plots/")
    print("="*70)

if __name__ == "__main__":
    run_evaluation()