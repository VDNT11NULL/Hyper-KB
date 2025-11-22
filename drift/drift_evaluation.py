"""
Evaluation metrics and benchmarks for drift handling.
"""

from typing import List, Dict, Tuple
import numpy as np
from drift import DriftDetector, DriftAnalyzer


class DriftEvaluator:
    """
    Evaluates drift handling performance.
    """
    
    def __init__(self):
        self.analyzer = DriftAnalyzer()
    
    def evaluate_shift_detection(
        self,
        predicted_shifts: List[bool],
        ground_truth_shifts: List[bool]
    ) -> Dict[str, float]:
        """
        Evaluate shift detection accuracy.
        
        Args:
            predicted_shifts: Predicted shift indicators
            ground_truth_shifts: Ground truth indicators
            
        Returns:
            Metrics dict with accuracy, precision, recall, F1
        """
        if len(predicted_shifts) != len(ground_truth_shifts):
            raise ValueError("Predictions and ground truth must match in length")
        
        tp = sum(p and g for p, g in zip(predicted_shifts, ground_truth_shifts))
        fp = sum(p and not g for p, g in zip(predicted_shifts, ground_truth_shifts))
        fn = sum(not p and g for p, g in zip(predicted_shifts, ground_truth_shifts))
        tn = sum(not p and not g for p, g in zip(predicted_shifts, ground_truth_shifts))
        
        total = len(predicted_shifts)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        }
    
    def evaluate_context_maintenance(
        self,
        retrieved_contexts_per_turn: List[List[str]],
        relevant_contexts_per_turn: List[List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate how well system maintains relevant context.
        
        Args:
            retrieved_contexts_per_turn: Retrieved context IDs per turn
            relevant_contexts_per_turn: Relevant context IDs per turn
            
        Returns:
            Metrics including precision, recall, stability
        """
        precisions = []
        recalls = []
        
        for retrieved, relevant in zip(retrieved_contexts_per_turn, relevant_contexts_per_turn):
            retrieved_set = set(retrieved)
            relevant_set = set(relevant)
            
            if retrieved_set:
                precision = len(retrieved_set & relevant_set) / len(retrieved_set)
                precisions.append(precision)
            
            if relevant_set:
                recall = len(retrieved_set & relevant_set) / len(relevant_set)
                recalls.append(recall)
        
        # Compute stability (overlap between consecutive retrievals)
        stability_scores = []
        for i in range(len(retrieved_contexts_per_turn) - 1):
            current = set(retrieved_contexts_per_turn[i])
            next_set = set(retrieved_contexts_per_turn[i + 1])
            
            if current or next_set:
                overlap = len(current & next_set) / len(current | next_set)
                stability_scores.append(overlap)
        
        return {
            'avg_precision': np.mean(precisions) if precisions else 0.0,
            'avg_recall': np.mean(recalls) if recalls else 0.0,
            'context_stability': np.mean(stability_scores) if stability_scores else 0.0
        }