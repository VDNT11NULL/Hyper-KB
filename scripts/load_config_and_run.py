"""
Load experiment configuration from YAML and execute.
Enables reproducible experiments from configuration files.
"""

import yaml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kb.kb_enhanced import EnhancedKnowledgeBase
from retrieval.sparse_retrieval import BM25Retriever, MongoFTSRetriever
from retrieval.dense_retrieval import FAISSRetriever
from retrieval.hybrid_fusion import HybridRetriever
from evaluation.benchmarks import SyntheticBenchmarkGenerator, BenchmarkRunner
from evaluation.visualization import ExperimentVisualizer
from evaluation.experiment_tracker import ExperimentTracker


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_retriever_from_config(config: dict, kb: EnhancedKnowledgeBase):
    """Create retriever instance from configuration."""
    retriever_config = config['retriever']
    retriever_type = retriever_config['type']
    
    if retriever_type == 'sparse':
        method = retriever_config.get('sparse_method', 'bm25')
        params = retriever_config.get('sparse_params', {})
        
        if method == 'bm25':
            retriever = BM25Retriever(
                kb,
                k1=params.get('k1', 1.5),
                b=params.get('b', 0.75)
            )
        elif method == 'mongo_fts':
            retriever = MongoFTSRetriever(kb)
        else:
            raise ValueError(f"Unknown sparse method: {method}")
        
        retriever.index_documents()
        return retriever
    
    elif retriever_type == 'dense':
        method = retriever_config.get('dense_method', 'faiss')
        params = retriever_config.get('dense_params', {})
        
        if method == 'faiss':
            retriever = FAISSRetriever(
                kb,
                dimension=params.get('dimension', 384),
                index_type=params.get('index_type', 'flat')
            )
        else:
            raise ValueError(f"Unknown dense method: {method}")
        
        retriever.index_embeddings()
        return retriever
    
    elif retriever_type == 'hybrid':
        sparse_method = retriever_config.get('sparse_method', 'bm25')
        dense_method = retriever_config.get('dense_method', 'faiss')
        fusion_method = retriever_config.get('fusion_method', 'rrf')
        
        sparse_params = retriever_config.get('sparse_params', {})
        dense_params = retriever_config.get('dense_params', {})
        fusion_params = retriever_config.get('fusion_params', {})
        
        # Create sparse retriever
        if sparse_method == 'bm25':
            sparse = BM25Retriever(
                kb,
                k1=sparse_params.get('k1', 1.5),
                b=sparse_params.get('b', 0.75)
            )
        elif sparse_method == 'mongo_fts':
            sparse = MongoFTSRetriever(kb)
        else:
            raise ValueError(f"Unknown sparse method: {sparse_method}")
        
        sparse.index_documents()
        
        # Create dense retriever
        if dense_method == 'faiss':
            dense = FAISSRetriever(
                kb,
                dimension=dense_params.get('dimension', 384),
                index_type=dense_params.get('index_type', 'flat')
            )
        else:
            raise ValueError(f"Unknown dense method: {dense_method}")
        
        dense.index_embeddings()
        
        # Create hybrid retriever
        retriever = HybridRetriever(
            sparse_retriever=sparse,
            dense_retriever=dense,
            fusion_method=fusion_method,
            fusion_params=fusion_params
        )
        
        return retriever
    
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


def create_dataset_from_config(config: dict):
    """Create benchmark dataset from configuration."""
    dataset_config = config['dataset']
    generator = dataset_config.get('generator', 'synthetic')
    params = dataset_config.get('params', {})
    
    if generator == 'synthetic':
        dataset = SyntheticBenchmarkGenerator.generate_conversational_dataset(
            num_sessions=params.get('num_sessions', 10),
            turns_per_session=tuple(params.get('turns_per_session', [3, 8])),
            topic_shift_probability=params.get('topic_shift_probability', 0.2)
        )
        return dataset
    else:
        raise ValueError(f"Unknown dataset generator: {generator}")


def run_experiment_from_config(config_path: str):
    """Run complete experiment from configuration file."""
    print("=" * 70)
    print("CONFIGURATION-BASED EXPERIMENT RUNNER")
    print("=" * 70)
    
    # Load configuration
    print(f"\nLoading configuration from: {config_path}")
    config = load_config(config_path)
    
    exp_config = config['experiment']
    print(f"Experiment: {exp_config['name']}")
    print(f"Description: {exp_config['description']}")
    
    # Initialize KB
    print("\nInitializing Knowledge Base...")
    kb = EnhancedKnowledgeBase(db_name="hyper_kb_config_exp")
    
    # Create retriever
    print("\nCreating retriever...")
    retriever = create_retriever_from_config(config, kb)
    print(f"Retriever type: {config['retriever']['type']}")
    
    # Create dataset
    print("\nCreating benchmark dataset...")
    dataset = create_dataset_from_config(config)
    print(f"Dataset: {dataset.name} ({len(dataset.queries)} queries)")
    
    # Initialize tracking
    tracker = ExperimentTracker()
    
    # Create experiment
    exp_track_config = tracker.create_experiment(
        experiment_name=exp_config['name'],
        description=exp_config['description'],
        retriever_type=config['retriever']['type'],
        retriever_params=config['retriever'],
        dataset_name=dataset.name,
        dataset_params=config['dataset']['params'],
        evaluation_metrics=config['evaluation']['metrics'],
        tags=exp_config.get('tags', [])
    )
    
    # Run benchmark
    print("\nRunning benchmark...")
    runner = BenchmarkRunner()
    result = runner.run_retrieval_benchmark(
        retriever=retriever,
        dataset=dataset,
        retriever_name=exp_config['name'],
        top_k_values=config['evaluation']['top_k_values'],
        measure_latency=config['evaluation']['measure_latency']
    )
    
    # Log results
    tracker.log_result(
        experiment_id=exp_track_config.experiment_id,
        metrics=result['metrics'],
        latency_stats=result.get('latency')
    )
    
    # Generate visualizations if requested
    if config['output'].get('generate_plots', True):
        print("\nGenerating visualizations...")
        visualizer = ExperimentVisualizer()
        
        # Create single-experiment comparison structure
        comparison = {
            'dataset': dataset.name,
            'timestamp': result['timestamp'],
            'retrievers': [result]
        }
        
        visualizer.generate_all_comparison_plots(
            comparison,
            k_values=config['evaluation']['top_k_values']
        )
    
    # Cleanup
    kb.close()
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"\nExperiment ID: {exp_track_config.experiment_id}")
    print(f"Results saved to: experiments/results/")


def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        print("Usage: python load_config_and_run.py <config_file.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    run_experiment_from_config(config_path)


if __name__ == "__main__":
    main()