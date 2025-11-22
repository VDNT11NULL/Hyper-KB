"""
FIXED: Complete evaluation with plots for report.
Uses real KB data with actual interaction IDs for accurate metrics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.end_to_end import HybridRetrievalPipeline
from evaluation.benchmarks import SyntheticBenchmarkGenerator, BenchmarkRunner
from evaluation.visualization import ExperimentVisualizer
from sentence_transformers import SentenceTransformer
import json

def main():
    print("="*70)
    print("RUNNING COMPLETE EVALUATION (FIXED)")
    print("="*70)
    
    # Initialize
    pipeline = HybridRetrievalPipeline(db_name="eval_benchmark")
    
    # Load embedding model
    print("\n[0/5] Loading embedding model...")
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Loaded embedding model: all-MiniLM-L6-v2")
    except Exception as e:
        print(f"✗ Failed to load embedding model: {e}")
        return
    
    # Seed data
    print("\n[1/5] Seeding evaluation data...")
    from scripts.seed_data import seed_database
    seed_database(db_name="eval_benchmark", num_sessions=5)
    
    # Generate benchmark dataset FROM REAL KB DATA
    print("\n[2/5] Generating benchmark dataset from real KB data...")
    dataset = SyntheticBenchmarkGenerator.generate_conversational_dataset_from_kb(
        kb=pipeline.kb,
        num_sessions=None,  # Use all seeded sessions
        min_turns_for_query=2
    )
    
    if not dataset.queries:
        print("✗ No queries generated! Check your seeded data.")
        pipeline.close()
        return
    
    print(f"\n✓ Generated {len(dataset.queries)} test queries")
    print(f"  From {dataset.metadata['num_sessions']} sessions")
    
    # Run benchmarks
    print("\n[3/5] Running benchmarks...")
    runner = BenchmarkRunner(
        output_dir="experiments/results",
        embedding_model=embedding_model
    )
    
    # Test different retrievers
    from retrieval.sparse_retrieval import BM25Retriever
    from retrieval.dense_retrieval import FAISSRetriever
    from retrieval.hybrid_fusion import HybridRetriever
    
    print("\nIndexing retrievers...")
    sparse = BM25Retriever(pipeline.kb)
    sparse.index_documents()
    
    dense = FAISSRetriever(pipeline.kb, dimension=384)
    dense.index_embeddings()
    
    hybrid = HybridRetriever(sparse, dense, fusion_method='rrf')
    
    retrievers = {
        'BM25': sparse,
        'FAISS': dense,
        'Hybrid_RRF': hybrid
    }
    
    print("\nRunning comparison benchmark...")
    results = runner.run_comparison_benchmark(
        retrievers=retrievers,
        dataset=dataset,
        top_k_values=[1, 3, 5, 10]
    )
    
    # Generate plots
    print("\n[4/5] Generating visualizations...")
    try:
        visualizer = ExperimentVisualizer(output_dir="experiments/results/plots")
        visualizer.generate_all_comparison_plots(results, k_values=[1, 3, 5, 10])
        print("✓ Visualizations generated")
    except Exception as e:
        print(f"⚠ Warning: Could not generate plots: {e}")
    
    # Save summary
    print("\n[5/5] Saving summary...")
    summary = {
        'total_queries': len(dataset.queries),
        'methods_compared': list(retrievers.keys()),
        'results_summary': {
            name: result['metrics'] 
            for name, result in zip(retrievers.keys(), results['retrievers'])
        }
    }
    
    output_file = Path("experiments/results/evaluation_summary.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: experiments/results/")
    print(f"Plots saved to: experiments/results/plots/")
    print(f"Summary: {output_file}")
    
    # Print detailed summary
    print("\n### RESULTS SUMMARY ###")
    for name in retrievers.keys():
        metrics = summary['results_summary'][name]
        print(f"\n{name}:")
        print(f"  MAP:    {metrics.get('MAP', 0):.4f}")
        print(f"  MRR:    {metrics.get('MRR', 0):.4f}")
        print(f"  P@1:    {metrics.get('P@1', 0):.4f}")
        print(f"  P@5:    {metrics.get('P@5', 0):.4f}")
        print(f"  R@5:    {metrics.get('R@5', 0):.4f}")
        print(f"  nDCG@5: {metrics.get('nDCG@5', 0):.4f}")
    
    # Check if metrics are still zero
    all_metrics = [m for result in summary['results_summary'].values() for m in result.values()]
    if all(m == 0 for m in all_metrics):
        print("\n⚠ WARNING: All metrics are still zero!")
        print("  This suggests the retrievers are not finding the relevant documents.")
        print("  Possible issues:")
        print("  1. Retriever indexes may not include the seeded interactions")
        print("  2. Ground truth IDs don't match retrieved IDs")
        print("  3. Retrievers are returning empty results")
    else:
        print("\n✓ Metrics look good! Evaluation successful.")
    
    pipeline.close()

if __name__ == "__main__":
    main()