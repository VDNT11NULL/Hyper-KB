"""
Complete benchmark across entire pipeline.
Tests all components end-to-end.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from uuid import uuid4
from datetime import datetime

from pipeline import HybridRetrievalPipeline
from evaluation.benchmarks import SyntheticBenchmarkGenerator, BenchmarkRunner
from evaluation.experiment_tracker import ExperimentTracker
from evaluation.visualization import ExperimentVisualizer
from evaluation.drift_evaluation import DriftEvaluator


def main():
    print("=" * 70)
    print("COMPLETE END-TO-END BENCHMARK")
    print("=" * 70)
    
    # Initialize
    pipeline = HybridRetrievalPipeline(db_name="hyper_kb_benchmark")
    tracker = ExperimentTracker()
    visualizer = ExperimentVisualizer()
    
    # Generate test dataset with drift
    print("\n[1] Generating benchmark dataset...")
    dataset = SyntheticBenchmarkGenerator.generate_conversational_dataset(
        num_sessions=5,
        turns_per_session=(4, 8),
        topic_shift_probability=0.3
    )
    print(f"Generated {len(dataset.queries)} queries across {5} sessions")
    
    # Seed pipeline with data
    print("\n[2] Seeding pipeline...")
    sessions = {}
    for query_obj in dataset.queries:
        if query_obj.session_id not in sessions:
            sessions[query_obj.session_id] = []
        sessions[query_obj.session_id].append(query_obj)
    
    for session_id, queries in sessions.items():
        for q in queries:
            # Simulate storing interaction
            pipeline.process_interaction(
                query=q.query_text,
                response=f"Response to: {q.query_text}",
                session_id=session_id
            )
    
    # Test configurations
    configs = [
        {
            'name': 'Standard_Hybrid',
            'fusion': 'rrf',
            'aggregation': 'concatenate',
            'adaptive': False
        },
        {
            'name': 'Weighted_Adaptive',
            'fusion': 'weighted',
            'aggregation': 'weighted',
            'adaptive': True
        },
        {
            'name': 'Distribution_Recency',
            'fusion': 'distribution',
            'aggregation': 'recency',
            'adaptive': True
        }
    ]
    
    results = []
    
    print("\n[3] Running benchmarks...")
    for config in configs:
        print(f"\n  Testing: {config['name']}")
        
        # Create experiment
        exp_config = tracker.create_experiment(
            experiment_name=config['name'],
            description=f"End-to-end test with {config['fusion']} fusion and {config['aggregation']} aggregation",
            retriever_type='hybrid',
            retriever_params=config,
            dataset_name=dataset.name,
            tags=['end-to-end', 'complete']
        )
        
        # Reinitialize pipeline with config
        test_pipeline = HybridRetrievalPipeline(
            db_name="hyper_kb_benchmark",
            fusion_method=config['fusion'],
            aggregation_strategy=config['aggregation']
        )
        test_pipeline.index_knowledge_base()
        
        # Run queries and measure
        query_results = []
        shift_predictions = []
        ground_truth_shifts = []
        
        for query_obj in dataset.queries:
            result = test_pipeline.query(
                query=query_obj.query_text,
                session_id=query_obj.session_id,
                top_k=5,
                use_adaptive=config['adaptive']
            )
            
            query_results.append(result)
            
            # Track shift detection
            shift_predictions.append(
                result['drift_state']['state']['last_shift_turn'] == result['session_turn']
                if result['drift_state']['state'] else False
            )
            ground_truth_shifts.append(query_obj.expected_topic_shift)
        
        # Evaluate drift detection
        drift_eval = DriftEvaluator()
        drift_metrics = drift_eval.evaluate_shift_detection(
            shift_predictions,
            ground_truth_shifts
        )
        
        print(f"    Drift Detection - Accuracy: {drift_metrics['accuracy']:.3f}, F1: {drift_metrics['f1']:.3f}")
        
        # Store results
        tracker.log_result(
            experiment_id=exp_config.experiment_id,
            metrics=drift_metrics,
            additional_data={
                'num_queries': len(query_results),
                'avg_retrieved_contexts': sum(r['retrieved_contexts'] for r in query_results) / len(query_results)
            }
        )
        
        results.append({
            'config': config,
            'metrics': drift_metrics,
            'experiment_id': exp_config.experiment_id
        })
        
        test_pipeline.close()
    
    # Generate comparison
    print("\n[4] Generating comparison report...")
    
    experiment_ids = [r['experiment_id'] for r in results]
    tracker.generate_comparison_report(
        experiment_ids=experiment_ids,
        output_file="experiments/results/end_to_end_comparison.md"
    )
    
    # Create comparison visualizations
    comparison_data = {
        'dataset': dataset.name,
        'timestamp': datetime.now().isoformat(),
        'retrievers': []
    }
    
    for r in results:
        comparison_data['retrievers'].append({
            'retriever_name': r['config']['name'],
            'metrics': r['metrics']
        })
    
    # Save comparison
    output_file = Path("experiments/results") / f"end_to_end_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Generate plots
    print("\n[5] Generating visualizations...")
    visualizer.generate_all_comparison_plots(comparison_data)
    
    pipeline.close()
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\nResults: experiments/results/")
    print(f"Plots: experiments/results/plots/")
    print(f"Report: experiments/results/end_to_end_comparison.md")


if __name__ == "__main__":
    main()