"""
End-to-End Pipeline Testing Script
Tests the complete Hyper-KB pipeline including:
- Data ingestion and curation
- Sparse/Dense/Hybrid retrieval
- Drift detection and adaptive retrieval
- Context aggregation and prompt enhancement
- Performance metrics and validation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
from uuid import uuid4
from datetime import datetime
from typing import Dict, List
import numpy as np

from pipeline.end_to_end import HybridRetrievalPipeline
from evaluation.metrics import RetrievalMetrics, LatencyMetrics, DriftMetrics


class PipelineE2ETest:
    """Comprehensive end-to-end pipeline test suite."""
    
    def __init__(self, db_name: str = "hyper_kb_e2e_test"):
        """Initialize test suite."""
        self.db_name = db_name
        self.pipeline = None
        self.test_session_id = None
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': []
        }
        
    def setup(self):
        """Setup test environment."""
        print("=" * 70)
        print("E2E PIPELINE TEST SUITE")
        print("=" * 70)
        print(f"\nInitializing test environment...")
        print(f"Database: {self.db_name}")
        
        try:
            self.pipeline = HybridRetrievalPipeline(
                db_name=self.db_name,
                fusion_method='rrf',
                aggregation_strategy='weighted',
                prompt_template='conversational'
            )
            self.test_session_id = str(uuid4())
            print("✓ Pipeline initialized successfully")
            return True
        except Exception as e:
            print(f"✗ Pipeline initialization failed: {e}")
            return False
    
    def teardown(self):
        """Cleanup test environment."""
        print("\n" + "=" * 70)
        print("CLEANUP")
        print("=" * 70)
        
        if self.pipeline:
            try:
                # Optional: Clear test data
                # self.pipeline.kb.interactions.delete_many({'session_id': self.test_session_id})
                self.pipeline.close()
                print("✓ Pipeline closed successfully")
            except Exception as e:
                print(f"⚠ Cleanup warning: {e}")
    
    def log_test_result(self, test_name: str, passed: bool, details: Dict = None):
        """Log individual test result."""
        if passed:
            self.test_results['tests_passed'] += 1
            status = "✓ PASSED"
        else:
            self.test_results['tests_failed'] += 1
            status = "✗ FAILED"
        
        print(f"  {status}: {test_name}")
        
        self.test_results['test_details'].append({
            'test_name': test_name,
            'passed': passed,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        })
    
    # ==================== TEST CASES ====================
    
    def test_01_data_ingestion(self):
        """Test data ingestion and curation."""
        print("\n[TEST 1] Data Ingestion and Curation")
        print("-" * 70)
        
        test_data = [
            {
                "query": "What is machine learning?",
                "response": "Machine learning is a subset of AI that enables systems to learn from data."
            },
            {
                "query": "Explain supervised learning",
                "response": "Supervised learning uses labeled data to train predictive models."
            },
            {
                "query": "What about deep learning?",
                "response": "Deep learning uses neural networks with multiple layers to learn hierarchical representations."
            }
        ]
        
        try:
            interaction_ids = []
            for data in test_data:
                result = self.pipeline.process_interaction(
                    query=data['query'],
                    response=data['response'],
                    session_id=self.test_session_id,
                    store=True
                )
                interaction_ids.append(result['interaction_id'])
            
            # Validate storage
            stored_count = self.pipeline.kb.interactions.count_documents({
                'session_id': self.test_session_id
            })
            
            passed = (stored_count == len(test_data)) and all(interaction_ids)
            self.log_test_result(
                "Data Ingestion",
                passed,
                {
                    'expected_count': len(test_data),
                    'stored_count': stored_count,
                    'interaction_ids': interaction_ids
                }
            )
            return passed
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            self.log_test_result("Data Ingestion", False, {'error': str(e)})
            return False
    
    def test_02_feature_extraction(self):
        """Test feature extraction (keywords, entities, embeddings)."""
        print("\n[TEST 2] Feature Extraction")
        print("-" * 70)
        
        try:
            # Get stored interactions
            interactions = list(self.pipeline.kb.interactions.find({
                'session_id': self.test_session_id
            }))
            
            if not interactions:
                self.log_test_result("Feature Extraction", False, {'error': 'No interactions found'})
                return False
            
            # Check features for each interaction
            features_found = 0
            embeddings_found = 0
            
            for interaction in interactions:
                interaction_id = interaction['interaction_id']
                
                # Check features collection
                feature = self.pipeline.kb.features.find_one({
                    'interaction_id': interaction_id
                })
                if feature and (feature.get('keywords') or feature.get('context_passage')):
                    features_found += 1
                
                # Check embeddings collection
                embedding = self.pipeline.kb.embeddings.find_one({
                    'interaction_id': interaction_id
                })
                if embedding and embedding.get('embedding_vector'):
                    embeddings_found += 1
            
            passed = (features_found == len(interactions)) and (embeddings_found == len(interactions))
            self.log_test_result(
                "Feature Extraction",
                passed,
                {
                    'total_interactions': len(interactions),
                    'features_found': features_found,
                    'embeddings_found': embeddings_found
                }
            )
            return passed
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            self.log_test_result("Feature Extraction", False, {'error': str(e)})
            return False
    
    def test_03_indexing(self):
        """Test retriever indexing."""
        print("\n[TEST 3] Retriever Indexing")
        print("-" * 70)
        
        try:
            # Ensure indexing
            self.pipeline.ensure_indexed()
            
            # Check if indexes are built
            sparse_indexed = (self.pipeline.sparse_retriever is not None and 
                            self.pipeline.sparse_retriever.bm25 is not None)
            
            dense_indexed = (self.pipeline.dense_retriever is not None and 
                           self.pipeline.dense_retriever.index is not None)
            
            hybrid_indexed = self.pipeline.hybrid_retriever is not None
            
            passed = sparse_indexed and dense_indexed and hybrid_indexed and self.pipeline.indexed
            
            self.log_test_result(
                "Retriever Indexing",
                passed,
                {
                    'sparse_indexed': sparse_indexed,
                    'dense_indexed': dense_indexed,
                    'hybrid_indexed': hybrid_indexed,
                    'pipeline_indexed_flag': self.pipeline.indexed
                }
            )
            return passed
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            self.log_test_result("Retriever Indexing", False, {'error': str(e)})
            return False
    
    def test_04_sparse_retrieval(self):
        """Test sparse (BM25) retrieval."""
        print("\n[TEST 4] Sparse Retrieval")
        print("-" * 70)
        
        try:
            test_query = "machine learning algorithms"
            query_embedding = self.pipeline.curator.get_embedding_for_query(test_query)
            
            # Use sparse only
            results = self.pipeline.hybrid_retriever.search_sparse_only(
                query=test_query,
                top_k=3
            )
            
            passed = len(results) > 0 and all(r.score > 0 for r in results)
            
            self.log_test_result(
                "Sparse Retrieval",
                passed,
                {
                    'query': test_query,
                    'results_count': len(results),
                    'top_score': results[0].score if results else 0,
                    'method': results[0].retrieval_method if results else None
                }
            )
            return passed
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            self.log_test_result("Sparse Retrieval", False, {'error': str(e)})
            return False
    
    def test_05_dense_retrieval(self):
        """Test dense (FAISS) retrieval."""
        print("\n[TEST 5] Dense Retrieval")
        print("-" * 70)
        
        try:
            test_query = "neural networks and deep learning"
            query_embedding = self.pipeline.curator.get_embedding_for_query(test_query)
            
            # Use dense only
            results = self.pipeline.hybrid_retriever.search_dense_only(
                query_embedding=query_embedding,
                top_k=3
            )
            
            passed = len(results) > 0 and all(r.score > 0 for r in results)
            
            self.log_test_result(
                "Dense Retrieval",
                passed,
                {
                    'query': test_query,
                    'results_count': len(results),
                    'top_score': results[0].score if results else 0,
                    'method': results[0].retrieval_method if results else None
                }
            )
            return passed
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            self.log_test_result("Dense Retrieval", False, {'error': str(e)})
            return False
    
    def test_06_hybrid_retrieval(self):
        """Test hybrid retrieval with fusion."""
        print("\n[TEST 6] Hybrid Retrieval")
        print("-" * 70)
        
        try:
            test_query = "supervised and unsupervised learning"
            query_embedding = self.pipeline.curator.get_embedding_for_query(test_query)
            
            # Use hybrid
            results = self.pipeline.hybrid_retriever.search(
                query=test_query,
                query_embedding=query_embedding,
                top_k=3
            )
            
            # Verify RRF fusion was used
            fusion_used = results[0].retrieval_method == 'hybrid_rrf' if results else False
            
            passed = len(results) > 0 and fusion_used and all(r.score > 0 for r in results)
            
            self.log_test_result(
                "Hybrid Retrieval",
                passed,
                {
                    'query': test_query,
                    'results_count': len(results),
                    'top_score': results[0].score if results else 0,
                    'fusion_method': results[0].retrieval_method if results else None
                }
            )
            return passed
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            self.log_test_result("Hybrid Retrieval", False, {'error': str(e)})
            return False
    
    def test_07_drift_detection(self):
        """Test topic drift detection."""
        print("\n[TEST 7] Topic Drift Detection")
        print("-" * 70)
        
        try:
            # Add interaction with topic shift
            shift_result = self.pipeline.process_interaction(
                query="How do I cook pasta?",  # Different topic
                response="Boil water, add salt, cook pasta for 8-10 minutes.",
                session_id=self.test_session_id,
                store=True
            )
            
            # Check if shift was detected
            shift_detected = shift_result.get('shift_detected', False)
            shift_score = shift_result.get('topic_shift_score', 0.0)
            
            passed = shift_detected and shift_score > 0.3
            
            self.log_test_result(
                "Drift Detection",
                passed,
                {
                    'shift_detected': shift_detected,
                    'shift_score': shift_score,
                    'expected_shift': True
                }
            )
            return passed
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            self.log_test_result("Drift Detection", False, {'error': str(e)})
            return False
    
    def test_08_context_tracking(self):
        """Test conversational context tracking."""
        print("\n[TEST 8] Context Tracking")
        print("-" * 70)
        
        try:
            # Check context tracker state
            drift_state = self.pipeline.context_tracker.get_session_summary(
                self.test_session_id
            )
            
            total_interactions = drift_state.get('total_interactions', 0)
            has_state = drift_state.get('state') is not None
            
            passed = total_interactions > 0 and has_state
            
            self.log_test_result(
                "Context Tracking",
                passed,
                {
                    'total_interactions': total_interactions,
                    'topic_transitions': drift_state.get('topic_transitions', 0),
                    'active_context_size': drift_state.get('active_context_size', 0)
                }
            )
            return passed
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            self.log_test_result("Context Tracking", False, {'error': str(e)})
            return False
    
    def test_09_context_aggregation(self):
        """Test context aggregation strategies."""
        print("\n[TEST 9] Context Aggregation")
        print("-" * 70)
        
        try:
            test_query = "explain machine learning concepts"
            
            result = self.pipeline.query(
                query=test_query,
                session_id=self.test_session_id,
                top_k=3
            )
            
            enhanced_prompt = result.get('enhanced_prompt', '')
            retrieved_contexts = result.get('retrieved_contexts', 0)
            
            # Verify prompt contains context
            has_context = len(enhanced_prompt) > len(test_query)
            has_results = retrieved_contexts > 0
            
            passed = has_context and has_results
            
            self.log_test_result(
                "Context Aggregation",
                passed,
                {
                    'retrieved_contexts': retrieved_contexts,
                    'prompt_length': len(enhanced_prompt),
                    'contains_context': has_context
                }
            )
            return passed
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            self.log_test_result("Context Aggregation", False, {'error': str(e)})
            return False
    
    def test_10_prompt_enhancement(self):
        """Test prompt building and enhancement."""
        print("\n[TEST 10] Prompt Enhancement")
        print("-" * 70)
        
        try:
            test_query = "what is supervised learning"
            
            result = self.pipeline.query(
                query=test_query,
                session_id=self.test_session_id,
                top_k=3
            )
            
            enhanced_prompt = result.get('enhanced_prompt', '')
            
            # Check prompt structure
            has_system_prompt = 'helpful AI assistant' in enhanced_prompt.lower()
            has_context = 'context' in enhanced_prompt.lower()
            has_query = test_query in enhanced_prompt.lower()
            
            passed = has_system_prompt and has_context and has_query
            
            self.log_test_result(
                "Prompt Enhancement",
                passed,
                {
                    'prompt_length': len(enhanced_prompt),
                    'has_system_prompt': has_system_prompt,
                    'has_context_section': has_context,
                    'has_query': has_query
                }
            )
            return passed
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            self.log_test_result("Prompt Enhancement", False, {'error': str(e)})
            return False
    
    def test_11_adaptive_retrieval(self):
        """Test adaptive retrieval based on drift."""
        print("\n[TEST 11] Adaptive Retrieval")
        print("-" * 70)
        
        try:
            # Query after topic shift
            test_query = "best pasta cooking techniques"
            
            result = self.pipeline.query(
                query=test_query,
                session_id=self.test_session_id,
                top_k=3,
                use_adaptive=True
            )
            
            retrieval_results = result.get('retrieval_results', [])
            drift_state = result.get('drift_state', {})
            
            # Check if adaptive retrieval adjusted strategy
            has_bias = 'retrieval_bias' in drift_state
            has_results = len(retrieval_results) > 0
            
            passed = has_results and has_bias
            
            self.log_test_result(
                "Adaptive Retrieval",
                passed,
                {
                    'retrieved_count': len(retrieval_results),
                    'retrieval_bias': drift_state.get('retrieval_bias', {}),
                    'adaptive_enabled': True
                }
            )
            return passed
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            self.log_test_result("Adaptive Retrieval", False, {'error': str(e)})
            return False
    
    def test_12_session_history(self):
        """Test session history retrieval."""
        print("\n[TEST 12] Session History")
        print("-" * 70)
        
        try:
            history = self.pipeline.kb.get_session_history(
                self.test_session_id,
                limit=10
            )
            
            # Verify chronological order
            if len(history) > 1:
                timestamps = [h['metadata']['timestamp'] for h in history]
                is_ordered = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
            else:
                is_ordered = True
            
            passed = len(history) > 0 and is_ordered
            
            self.log_test_result(
                "Session History",
                passed,
                {
                    'history_length': len(history),
                    'chronologically_ordered': is_ordered,
                    'session_id': self.test_session_id
                }
            )
            return passed
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            self.log_test_result("Session History", False, {'error': str(e)})
            return False
    
    def test_13_performance_metrics(self):
        """Test performance and latency metrics."""
        print("\n[TEST 13] Performance Metrics")
        print("-" * 70)
        
        try:
            # Measure query latency
            test_query = "machine learning"
            latencies = []
            
            for _ in range(5):
                start = time.time()
                result = self.pipeline.query(
                    query=test_query,
                    session_id=self.test_session_id,
                    top_k=3
                )
                latency = time.time() - start
                latencies.append(latency)
            
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            
            # Acceptable latency threshold: 2 seconds
            passed = avg_latency < 2.0 and p95_latency < 3.0
            
            self.log_test_result(
                "Performance Metrics",
                passed,
                {
                    'avg_latency_ms': avg_latency * 1000,
                    'p95_latency_ms': p95_latency * 1000,
                    'num_queries': len(latencies)
                }
            )
            return passed
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            self.log_test_result("Performance Metrics", False, {'error': str(e)})
            return False
    
    def test_14_statistics(self):
        """Test pipeline statistics."""
        print("\n[TEST 14] Pipeline Statistics")
        print("-" * 70)
        
        try:
            stats = self.pipeline.get_statistics()
            
            kb_stats = stats.get('kb_stats', {})
            has_interactions = kb_stats.get('total_interactions', 0) > 0
            has_sessions = kb_stats.get('unique_sessions', 0) > 0
            is_indexed = stats.get('indexed', False)
            
            passed = has_interactions and has_sessions and is_indexed
            
            self.log_test_result(
                "Pipeline Statistics",
                passed,
                {
                    'total_interactions': kb_stats.get('total_interactions', 0),
                    'unique_sessions': kb_stats.get('unique_sessions', 0),
                    'indexed': is_indexed,
                    'active_sessions': stats.get('active_sessions', 0)
                }
            )
            return passed
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            self.log_test_result("Pipeline Statistics", False, {'error': str(e)})
            return False
    
    def test_15_error_handling(self):
        """Test error handling and edge cases."""
        print("\n[TEST 15] Error Handling")
        print("-" * 70)
        
        try:
            errors_handled = 0
            total_tests = 0
            
            # Test 1: Empty query
            total_tests += 1
            try:
                result = self.pipeline.query("", self.test_session_id, top_k=1)
                errors_handled += 1
            except:
                pass
            
            # Test 2: Invalid session ID
            total_tests += 1
            try:
                result = self.pipeline.query("test", "invalid_session", top_k=1)
                errors_handled += 1
            except:
                pass
            
            # Test 3: Negative top_k
            total_tests += 1
            try:
                result = self.pipeline.query("test", self.test_session_id, top_k=-1)
                # Should either handle or raise appropriately
                errors_handled += 1
            except:
                errors_handled += 1
            
            passed = errors_handled == total_tests
            
            self.log_test_result(
                "Error Handling",
                passed,
                {
                    'edge_cases_tested': total_tests,
                    'handled_correctly': errors_handled
                }
            )
            return passed
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            self.log_test_result("Error Handling", False, {'error': str(e)})
            return False
    
    # ==================== MAIN TEST RUNNER ====================
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        if not self.setup():
            print("\n✗ Setup failed. Aborting tests.")
            return False
        
        print("\nRunning test suite...")
        print("=" * 70)
        
        # Run all tests
        tests = [
            self.test_01_data_ingestion,
            self.test_02_feature_extraction,
            self.test_03_indexing,
            self.test_04_sparse_retrieval,
            self.test_05_dense_retrieval,
            self.test_06_hybrid_retrieval,
            self.test_07_drift_detection,
            self.test_08_context_tracking,
            self.test_09_context_aggregation,
            self.test_10_prompt_enhancement,
            self.test_11_adaptive_retrieval,
            self.test_12_session_history,
            self.test_13_performance_metrics,
            self.test_14_statistics,
            self.test_15_error_handling
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"  ✗ Test execution error: {e}")
                self.log_test_result(test.__name__, False, {'error': str(e)})
        
        # Generate summary
        self.print_summary()
        
        # Save results
        self.save_results()
        
        self.teardown()
        
        return self.test_results['tests_failed'] == 0
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        total = self.test_results['tests_passed'] + self.test_results['tests_failed']
        pass_rate = (self.test_results['tests_passed'] / total * 100) if total > 0 else 0
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {self.test_results['tests_passed']}")
        print(f"Failed: {self.test_results['tests_failed']}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.test_results['tests_failed'] > 0:
            print("\nFailed Tests:")
            for test in self.test_results['test_details']:
                if not test['passed']:
                    print(f"  - {test['test_name']}")
                    if 'error' in test['details']:
                        print(f"    Error: {test['details']['error']}")
    
    def save_results(self):
        """Save test results to file."""
        output_dir = Path("experiments/results/tests")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"e2e_test_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\n✓ Test results saved to: {output_file}")


def main():
    """Main execution."""
    tester = PipelineE2ETest()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
