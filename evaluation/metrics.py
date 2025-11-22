"""
Comprehensive evaluation metrics for retrieval systems.
Implements standard IR metrics: MAP, MRR, nDCG, Precision, Recall, F1.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict


class RetrievalMetrics:
    """
    Collection of information retrieval evaluation metrics.
    All metrics support both binary relevance and graded relevance.
    """
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Precision at K: Proportion of retrieved documents that are relevant.
        
        Args:
            retrieved: List of retrieved document IDs (ranked)
            relevant: Set of relevant document IDs
            k: Cutoff position
            
        Returns:
            Precision at K (0.0 to 1.0)
        """
        if k == 0 or len(retrieved) == 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
        return relevant_retrieved / k
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Recall at K: Proportion of relevant documents that are retrieved.
        
        Args:
            retrieved: List of retrieved document IDs (ranked)
            relevant: Set of relevant document IDs
            k: Cutoff position
            
        Returns:
            Recall at K (0.0 to 1.0)
        """
        if len(relevant) == 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
        return relevant_retrieved / len(relevant)
    
    @staticmethod
    def f1_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        F1 score at K: Harmonic mean of precision and recall.
        
        Args:
            retrieved: List of retrieved document IDs (ranked)
            relevant: Set of relevant document IDs
            k: Cutoff position
            
        Returns:
            F1 score at K (0.0 to 1.0)
        """
        precision = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
        """
        Average Precision: Mean of precision values at each relevant document position.
        
        Args:
            retrieved: List of retrieved document IDs (ranked)
            relevant: Set of relevant document IDs
            
        Returns:
            Average Precision (0.0 to 1.0)
        """
        if len(relevant) == 0:
            return 0.0
        
        precision_sum = 0.0
        num_relevant_found = 0
        
        for k, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                num_relevant_found += 1
                precision_at_k = num_relevant_found / k
                precision_sum += precision_at_k
        
        return precision_sum / len(relevant)
    
    @staticmethod
    def mean_average_precision(
        all_retrieved: List[List[str]],
        all_relevant: List[Set[str]]
    ) -> float:
        """
        Mean Average Precision (MAP): Mean of AP across all queries.
        
        Args:
            all_retrieved: List of retrieved document lists (one per query)
            all_relevant: List of relevant document sets (one per query)
            
        Returns:
            MAP score (0.0 to 1.0)
        """
        if len(all_retrieved) == 0:
            return 0.0
        
        ap_scores = [
            RetrievalMetrics.average_precision(retrieved, relevant)
            for retrieved, relevant in zip(all_retrieved, all_relevant)
        ]
        
        return np.mean(ap_scores)
    
    @staticmethod
    def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
        """
        Reciprocal Rank: 1 / rank of first relevant document.
        
        Args:
            retrieved: List of retrieved document IDs (ranked)
            relevant: Set of relevant document IDs
            
        Returns:
            Reciprocal Rank (0.0 to 1.0)
        """
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0
    
    @staticmethod
    def mean_reciprocal_rank(
        all_retrieved: List[List[str]],
        all_relevant: List[Set[str]]
    ) -> float:
        """
        Mean Reciprocal Rank (MRR): Mean of RR across all queries.
        
        Args:
            all_retrieved: List of retrieved document lists (one per query)
            all_relevant: List of relevant document sets (one per query)
            
        Returns:
            MRR score (0.0 to 1.0)
        """
        if len(all_retrieved) == 0:
            return 0.0
        
        rr_scores = [
            RetrievalMetrics.reciprocal_rank(retrieved, relevant)
            for retrieved, relevant in zip(all_retrieved, all_relevant)
        ]
        
        return np.mean(rr_scores)
    
    @staticmethod
    def dcg_at_k(
        retrieved: List[str],
        relevance_scores: Dict[str, float],
        k: int
    ) -> float:
        """
        Discounted Cumulative Gain at K.
        
        Args:
            retrieved: List of retrieved document IDs (ranked)
            relevance_scores: Dictionary mapping doc_id to relevance score
            k: Cutoff position
            
        Returns:
            DCG at K
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            relevance = relevance_scores.get(doc_id, 0.0)
            dcg += relevance / np.log2(i + 1)
        return dcg
    
    @staticmethod
    def ndcg_at_k(
        retrieved: List[str],
        relevance_scores: Dict[str, float],
        k: int
    ) -> float:
        """
        Normalized Discounted Cumulative Gain at K.
        
        Args:
            retrieved: List of retrieved document IDs (ranked)
            relevance_scores: Dictionary mapping doc_id to relevance score
            k: Cutoff position
            
        Returns:
            nDCG at K (0.0 to 1.0)
        """
        dcg = RetrievalMetrics.dcg_at_k(retrieved, relevance_scores, k)
        
        # Calculate ideal DCG (sort by relevance)
        sorted_docs = sorted(
            relevance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        ideal_retrieved = [doc_id for doc_id, _ in sorted_docs]
        idcg = RetrievalMetrics.dcg_at_k(ideal_retrieved, relevance_scores, k)
        
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def mean_ndcg_at_k(
        all_retrieved: List[List[str]],
        all_relevance_scores: List[Dict[str, float]],
        k: int
    ) -> float:
        """
        Mean nDCG at K across all queries.
        
        Args:
            all_retrieved: List of retrieved document lists (one per query)
            all_relevance_scores: List of relevance score dictionaries
            k: Cutoff position
            
        Returns:
            Mean nDCG at K (0.0 to 1.0)
        """
        if len(all_retrieved) == 0:
            return 0.0
        
        ndcg_scores = [
            RetrievalMetrics.ndcg_at_k(retrieved, relevance_scores, k)
            for retrieved, relevance_scores in zip(all_retrieved, all_relevance_scores)
        ]
        
        return np.mean(ndcg_scores)


class LatencyMetrics:
    """
    Metrics for measuring retrieval latency and throughput.
    """
    
    @staticmethod
    def compute_statistics(latencies: List[float]) -> Dict[str, float]:
        """
        Compute latency statistics.
        
        Args:
            latencies: List of latency measurements in seconds
            
        Returns:
            Dictionary with statistical measures
        """
        if not latencies:
            return {}
        
        latencies_array = np.array(latencies)
        
        return {
            'mean': float(np.mean(latencies_array)),
            'median': float(np.median(latencies_array)),
            'std': float(np.std(latencies_array)),
            'min': float(np.min(latencies_array)),
            'max': float(np.max(latencies_array)),
            'p50': float(np.percentile(latencies_array, 50)),
            'p90': float(np.percentile(latencies_array, 90)),
            'p95': float(np.percentile(latencies_array, 95)),
            'p99': float(np.percentile(latencies_array, 99))
        }
    
    @staticmethod
    def queries_per_second(num_queries: int, total_time: float) -> float:
        """
        Calculate throughput in queries per second.
        
        Args:
            num_queries: Number of queries processed
            total_time: Total time in seconds
            
        Returns:
            Queries per second
        """
        if total_time <= 0:
            return 0.0
        return num_queries / total_time


class DriftMetrics:
    """
    Metrics for evaluating conversational drift handling.
    """
    
    @staticmethod
    def context_stability_score(
        retrieved_contexts: List[List[str]],
        session_id: str
    ) -> float:
        """
        Measure how stable retrieved contexts are across conversation turns.
        Higher score means more consistent context retrieval.
        
        Args:
            retrieved_contexts: List of retrieved doc IDs per turn
            session_id: Session identifier
            
        Returns:
            Stability score (0.0 to 1.0)
        """
        if len(retrieved_contexts) < 2:
            return 1.0
        
        # Calculate overlap between consecutive retrievals
        overlaps = []
        for i in range(len(retrieved_contexts) - 1):
            current = set(retrieved_contexts[i])
            next_set = set(retrieved_contexts[i + 1])
            
            if len(current) == 0 or len(next_set) == 0:
                continue
            
            intersection = len(current & next_set)
            union = len(current | next_set)
            
            if union > 0:
                overlaps.append(intersection / union)
        
        if not overlaps:
            return 0.0
        
        return np.mean(overlaps)
    
    @staticmethod
    def topic_shift_detection_accuracy(
        predicted_shifts: List[bool],
        ground_truth_shifts: List[bool]
    ) -> Dict[str, float]:
        """
        Evaluate topic shift detection accuracy.
        
        Args:
            predicted_shifts: Predicted shift indicators
            ground_truth_shifts: Ground truth shift indicators
            
        Returns:
            Dictionary with accuracy, precision, recall, F1
        """
        if len(predicted_shifts) != len(ground_truth_shifts):
            raise ValueError("Predicted and ground truth must have same length")
        
        if len(predicted_shifts) == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        true_positives = sum(
            p and g for p, g in zip(predicted_shifts, ground_truth_shifts)
        )
        false_positives = sum(
            p and not g for p, g in zip(predicted_shifts, ground_truth_shifts)
        )
        false_negatives = sum(
            not p and g for p, g in zip(predicted_shifts, ground_truth_shifts)
        )
        true_negatives = sum(
            not p and not g for p, g in zip(predicted_shifts, ground_truth_shifts)
        )
        
        total = len(predicted_shifts)
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
        
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0 else 0.0
        )
        
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0 else 0.0
        )
        
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }