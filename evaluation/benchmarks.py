"""
FIXED: Benchmark dataset definitions and evaluation runners.
Supports synthetic and real conversation datasets.
"""

import json
import time
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
from .metrics import RetrievalMetrics, LatencyMetrics


@dataclass
class BenchmarkQuery:
    """Single benchmark query with ground truth."""
    query_id: str
    query_text: str
    relevant_docs: Set[str]
    relevance_scores: Dict[str, float]
    session_id: Optional[str] = None
    turn_number: Optional[int] = None
    expected_topic_shift: bool = False


@dataclass
class BenchmarkDataset:
    """Collection of benchmark queries."""
    name: str
    description: str
    queries: List[BenchmarkQuery]
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SyntheticBenchmarkGenerator:
    """
    Generate synthetic benchmark datasets for testing.
    """
    
    @staticmethod
    def generate_conversational_dataset_from_kb(
        kb,
        num_sessions: Optional[int] = None,
        min_turns_for_query: int = 2
    ) -> BenchmarkDataset:
        """
        Generate benchmark from ACTUAL KB data instead of fake IDs.
        Uses real interaction IDs so metrics work correctly.
        
        Args:
            kb: Knowledge base instance to pull real data from
            num_sessions: Number of sessions to evaluate (None = all)
            min_turns_for_query: Minimum conversation turns before creating test query
            
        Returns:
            BenchmarkDataset with real interaction IDs
        """
        # Get all sessions from KB
        session_query = {}
        if num_sessions:
            all_sessions = list(kb.sessions.find(session_query).limit(num_sessions))
        else:
            all_sessions = list(kb.sessions.find(session_query))
        
        print(f"Generating benchmark from {len(all_sessions)} sessions...")
        
        queries = []
        
        for session in all_sessions:
            session_id = session['session_id']
            
            # Get all interactions for this session
            interactions = list(kb.interactions.find(
                {'session_id': session_id}
            ).sort('metadata.turn_number', 1))
            
            if len(interactions) < min_turns_for_query:
                print(f"  Skipping session {session_id[:8]}... (only {len(interactions)} turns)")
                continue
            
            print(f"  Session {session_id[:8]}...: {len(interactions)} interactions")
            
            # Create queries from each interaction (except first turn)
            for i, interaction in enumerate(interactions):
                if i == 0:
                    # Skip first turn - no context yet
                    continue
                
                query_id = f"q_{session_id[:8]}_{i}"
                query_text = interaction['query_text']
                
                # Ground truth: ALL previous interactions in same session
                # These represent the conversational context
                relevant_docs = {
                    interactions[j]['interaction_id'] 
                    for j in range(i)  # All previous turns
                }
                
                # Generate graded relevance scores
                # More recent interactions = more relevant
                relevance_scores = {}
                for j in range(i):
                    doc_id = interactions[j]['interaction_id']
                    # Exponential decay: most recent = 1.0, older = lower
                    recency_factor = (i - j)
                    relevance = max(0.1, 1.0 - (recency_factor - 1) * 0.15)
                    relevance_scores[doc_id] = min(1.0, relevance)
                
                # Detect expected topic shift
                topic_shift = interaction.get('metadata', {}).get('topic_shift_score', 0.0)
                expected_topic_shift = topic_shift > 0.3
                
                query = BenchmarkQuery(
                    query_id=query_id,
                    query_text=query_text,
                    relevant_docs=relevant_docs,
                    relevance_scores=relevance_scores,
                    session_id=session_id,
                    turn_number=i,
                    expected_topic_shift=expected_topic_shift
                )
                
                queries.append(query)
        
        print(f"✓ Generated {len(queries)} test queries with real IDs")
        if queries:
            sample_query = queries[0]
            print(f"  Sample query ID: {sample_query.query_id}")
            print(f"  Sample relevant docs: {len(sample_query.relevant_docs)} docs")
            print(f"  First relevant doc ID: {list(sample_query.relevant_docs)[0]}")
        
        return BenchmarkDataset(
            name="synthetic_conversational",
            description=f"Real conversational dataset from {len(all_sessions)} sessions",
            queries=queries,
            metadata={
                'num_sessions': len(all_sessions),
                'total_queries': len(queries),
                'data_source': 'real_kb_interactions'
            }
        )


class BenchmarkRunner:
    """
    Run retrieval benchmarks with proper embedding handling.
    """
    
    def __init__(self, output_dir: str = "experiments/results", embedding_model=None):
        """
        Initialize benchmark runner.
        
        Args:
            output_dir: Directory to save results
            embedding_model: Sentence transformer model for generating embeddings
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        
        # Initialize embedding model if not provided
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✓ Loaded embedding model: all-MiniLM-L6-v2")
            except Exception as e:
                print(f"⚠ Warning: Could not load embedding model: {e}")
                print("  Dense retrieval will fail.")
    
    def run_retrieval_benchmark(
        self,
        retriever,
        dataset: BenchmarkDataset,
        retriever_name: str,
        top_k_values: List[int] = [1, 3, 5, 10],
        measure_latency: bool = True
    ) -> Dict:
        """
        Run comprehensive retrieval benchmark with proper API handling.
        """
        print(f"\nRunning benchmark: {dataset.name}")
        print(f"Retriever: {retriever_name}")
        print(f"Queries: {len(dataset.queries)}")
        print("-" * 70)
        
        all_retrieved = []
        all_relevant = []
        all_relevance_scores = []
        latencies = []
        
        # Determine retriever type
        retriever_type = type(retriever).__name__
        print(f"Detected retriever type: {retriever_type}")
        
        # Run queries
        for idx, query_obj in enumerate(dataset.queries):
            if measure_latency:
                start_time = time.time()
            
            try:
                # Handle different retriever APIs
                if retriever_type == 'BM25Retriever':
                    # BM25: needs query text only
                    results = retriever.search(
                        query=query_obj.query_text,
                        top_k=max(top_k_values)
                    )
                    
                elif retriever_type == 'FAISSRetriever':
                    # FAISS: needs query embedding
                    if self.embedding_model is None:
                        raise ValueError("Embedding model required for FAISS retrieval")
                    query_embedding = self.embedding_model.encode(query_obj.query_text)
                    results = retriever.search(
                        query_embedding=query_embedding,
                        top_k=max(top_k_values)
                    )
                    
                elif retriever_type == 'HybridRetriever':
                    # Hybrid: needs both query text and embedding
                    if self.embedding_model is None:
                        raise ValueError("Embedding model required for hybrid retrieval")
                    query_embedding = self.embedding_model.encode(query_obj.query_text)
                    results = retriever.search(
                        query=query_obj.query_text,
                        query_embedding=query_embedding,
                        top_k=max(top_k_values)
                    )
                    
                else:
                    # Generic fallback - try with query text
                    results = retriever.search(
                        query=query_obj.query_text,
                        top_k=max(top_k_values)
                    )
                
                retrieved_ids = [r.interaction_id for r in results]
                
                # Debug first query
                if idx == 0:
                    print(f"\nDebug info for first query:")
                    print(f"  Query: {query_obj.query_text[:50]}...")
                    print(f"  Retrieved {len(retrieved_ids)} docs")
                    print(f"  Sample retrieved ID: {retrieved_ids[0] if retrieved_ids else 'None'}")
                    print(f"  Expected {len(query_obj.relevant_docs)} relevant docs")
                    print(f"  Sample relevant ID: {list(query_obj.relevant_docs)[0] if query_obj.relevant_docs else 'None'}")
                    
                    # Check for overlap
                    overlap = set(retrieved_ids) & query_obj.relevant_docs
                    print(f"  Overlap: {len(overlap)} docs match ground truth")
                
            except Exception as e:
                print(f"Error retrieving for query {query_obj.query_id}: {e}")
                retrieved_ids = []
            
            if measure_latency:
                latency = time.time() - start_time
                latencies.append(latency)
            
            all_retrieved.append(retrieved_ids)
            all_relevant.append(query_obj.relevant_docs)
            all_relevance_scores.append(query_obj.relevance_scores)
        
        # Calculate metrics
        results = {
            'retriever_name': retriever_name,
            'dataset_name': dataset.name,
            'num_queries': len(dataset.queries),
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        # Calculate metrics at different K values
        for k in top_k_values:
            precision_scores = [
                RetrievalMetrics.precision_at_k(retrieved, relevant, k)
                for retrieved, relevant in zip(all_retrieved, all_relevant)
            ]
            
            recall_scores = [
                RetrievalMetrics.recall_at_k(retrieved, relevant, k)
                for retrieved, relevant in zip(all_retrieved, all_relevant)
            ]
            
            f1_scores = [
                RetrievalMetrics.f1_at_k(retrieved, relevant, k)
                for retrieved, relevant in zip(all_retrieved, all_relevant)
            ]
            
            ndcg_score = RetrievalMetrics.mean_ndcg_at_k(
                all_retrieved, all_relevance_scores, k
            )
            
            results['metrics'][f'P@{k}'] = float(np.mean(precision_scores))
            results['metrics'][f'R@{k}'] = float(np.mean(recall_scores))
            results['metrics'][f'F1@{k}'] = float(np.mean(f1_scores))
            results['metrics'][f'nDCG@{k}'] = float(ndcg_score)
        
        # MAP and MRR
        results['metrics']['MAP'] = RetrievalMetrics.mean_average_precision(
            all_retrieved, all_relevant
        )
        results['metrics']['MRR'] = RetrievalMetrics.mean_reciprocal_rank(
            all_retrieved, all_relevant
        )
        
        # Latency metrics
        if measure_latency and latencies:
            latency_stats = LatencyMetrics.compute_statistics(latencies)
            results['latency'] = latency_stats
            results['throughput_qps'] = LatencyMetrics.queries_per_second(
                len(dataset.queries), sum(latencies)
            )
        
        # Print results
        print("\nResults:")
        for metric_name, value in results['metrics'].items():
            print(f"  {metric_name}: {value:.4f}")
        
        if 'latency' in results:
            print(f"\nLatency (ms):")
            print(f"  Mean: {results['latency']['mean']*1000:.2f}")
            print(f"  P95: {results['latency']['p95']*1000:.2f}")
            print(f"  P99: {results['latency']['p99']*1000:.2f}")
            print(f"  Throughput: {results['throughput_qps']:.2f} queries/sec")
        
        # Save results
        self._save_results(results, retriever_name, dataset.name)
        
        return results
    
    def run_comparison_benchmark(
        self,
        retrievers: Dict[str, any],
        dataset: BenchmarkDataset,
        top_k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict:
        """
        Compare multiple retrievers on the same dataset.
        """
        comparison_results = {
            'dataset': dataset.name,
            'timestamp': datetime.now().isoformat(),
            'retrievers': []
        }
        
        for retriever_name, retriever in retrievers.items():
            result = self.run_retrieval_benchmark(
                retriever=retriever,
                dataset=dataset,
                retriever_name=retriever_name,
                top_k_values=top_k_values
            )
            comparison_results['retrievers'].append(result)
        
        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{dataset.name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"\nComparison results saved to: {filepath}")
        
        return comparison_results
    
    def _save_results(self, results: Dict, retriever_name: str, dataset_name: str):
        """Save individual benchmark results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{retriever_name}_{dataset_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")