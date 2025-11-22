"""
Visualization utilities for experiment results.
Generates publication-ready plots for reports.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict


# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class ExperimentVisualizer:
    """
    Create visualizations for experiment results.
    All plots are saved in proper format for LaTeX reports.
    """
    
    def __init__(self, output_dir: str = "experiments/results/plots"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_metric_comparison(
        self,
        results: Dict,
        metric_name: str,
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ):
        """
        Plot comparison of a single metric across retrievers.
        
        Args:
            results: Comparison results dictionary
            metric_name: Metric to plot (e.g., 'MAP', 'nDCG@10')
            title: Plot title
            save_name: Filename for saving
        """
        retrievers = []
        values = []
        
        for retriever_result in results['retrievers']:
            retrievers.append(retriever_result['retriever_name'])
            metric_value = retriever_result['metrics'].get(metric_name, 0.0)
            values.append(metric_value)
        
        fig, ax = plt.subplots()
        
        bars = ax.bar(retrievers, values, color=sns.color_palette("husl", len(retrievers)))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=12
            )
        
        ax.set_ylabel(metric_name, fontsize=14)
        ax.set_xlabel('Retriever', fontsize=14)
        
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold')
        else:
            ax.set_title(f'{metric_name} Comparison', fontsize=16, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_name is None:
            save_name = f"comparison_{metric_name.replace('@', '_at_')}.png"
        
        filepath = self.output_dir / save_name
        plt.savefig(filepath)
        print(f"Saved plot: {filepath}")
        plt.close()
    
    def plot_all_metrics_comparison(
        self,
        results: Dict,
        metrics: List[str] = None,
        save_name: str = "all_metrics_comparison.png"
    ):
        """
        Plot grouped bar chart comparing all metrics across retrievers.
        
        Args:
            results: Comparison results dictionary
            metrics: List of metrics to plot (if None, uses all)
            save_name: Filename for saving
        """
        if metrics is None:
            # Get all metrics from first retriever
            first_retriever = results['retrievers'][0]
            metrics = list(first_retriever['metrics'].keys())
        
        retrievers = [r['retriever_name'] for r in results['retrievers']]
        
        # Organize data
        data = {metric: [] for metric in metrics}
        for retriever_result in results['retrievers']:
            for metric in metrics:
                value = retriever_result['metrics'].get(metric, 0.0)
                data[metric].append(value)
        
        # Plot
        x = np.arange(len(retrievers))
        width = 0.8 / len(metrics)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = sns.color_palette("husl", len(metrics))
        
        for i, metric in enumerate(metrics):
            offset = (i - len(metrics)/2 + 0.5) * width
            ax.bar(x + offset, data[metric], width, label=metric, color=colors[i])
        
        ax.set_ylabel('Score', fontsize=14)
        ax.set_xlabel('Retriever', fontsize=14)
        ax.set_title('Retrieval Metrics Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(retrievers, rotation=45, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        filepath = self.output_dir / save_name
        plt.savefig(filepath)
        print(f"Saved plot: {filepath}")
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        results: Dict,
        k_values: List[int],
        save_name: str = "precision_recall_curve.png"
    ):
        """
        Plot precision-recall curves for different retrievers.
        
        Args:
            results: Comparison results dictionary
            k_values: List of K values
            save_name: Filename for saving
        """
        fig, ax = plt.subplots()
        
        colors = sns.color_palette("husl", len(results['retrievers']))
        
        for idx, retriever_result in enumerate(results['retrievers']):
            retriever_name = retriever_result['retriever_name']
            
            precisions = []
            recalls = []
            
            for k in k_values:
                precision = retriever_result['metrics'].get(f'P@{k}', 0.0)
                recall = retriever_result['metrics'].get(f'R@{k}', 0.0)
                precisions.append(precision)
                recalls.append(recall)
            
            ax.plot(
                recalls, precisions,
                marker='o', linewidth=2, markersize=8,
                label=retriever_name, color=colors[idx]
            )
        
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title('Precision-Recall Curves', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        filepath = self.output_dir / save_name
        plt.savefig(filepath)
        print(f"Saved plot: {filepath}")
        plt.close()
    
    def plot_latency_comparison(
        self,
        results: Dict,
        save_name: str = "latency_comparison.png"
    ):
        """
        Plot latency statistics comparison across retrievers.
        
        Args:
            results: Comparison results dictionary
            save_name: Filename for saving
        """
        retrievers = []
        mean_latencies = []
        p95_latencies = []
        p99_latencies = []
        
        for retriever_result in results['retrievers']:
            if 'latency' not in retriever_result:
                continue
            
            retrievers.append(retriever_result['retriever_name'])
            latency = retriever_result['latency']
            mean_latencies.append(latency['mean'] * 1000)  # Convert to ms
            p95_latencies.append(latency['p95'] * 1000)
            p99_latencies.append(latency['p99'] * 1000)
        
        if not retrievers:
            print("No latency data available")
            return
        
        x = np.arange(len(retrievers))
        width = 0.25
        
        fig, ax = plt.subplots()
        
        ax.bar(x - width, mean_latencies, width, label='Mean', color='skyblue')
        ax.bar(x, p95_latencies, width, label='P95', color='lightcoral')
        ax.bar(x + width, p99_latencies, width, label='P99', color='lightgreen')
        
        ax.set_ylabel('Latency (ms)', fontsize=14)
        ax.set_xlabel('Retriever', fontsize=14)
        ax.set_title('Latency Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(retrievers, rotation=45, ha='right')
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        
        filepath = self.output_dir / save_name
        plt.savefig(filepath)
        print(f"Saved plot: {filepath}")
        plt.close()
    
    def plot_ndcg_at_k(
        self,
        results: Dict,
        k_values: List[int],
        save_name: str = "ndcg_at_k.png"
    ):
        """
        Plot nDCG@K for different K values across retrievers.
        
        Args:
            results: Comparison results dictionary
            k_values: List of K values
            save_name: Filename for saving
        """
        fig, ax = plt.subplots()
        
        colors = sns.color_palette("husl", len(results['retrievers']))
        
        for idx, retriever_result in enumerate(results['retrievers']):
            retriever_name = retriever_result['retriever_name']
            
            ndcg_values = []
            for k in k_values:
                ndcg = retriever_result['metrics'].get(f'nDCG@{k}', 0.0)
                ndcg_values.append(ndcg)
            
            ax.plot(
                k_values, ndcg_values,
                marker='o', linewidth=2, markersize=8,
                label=retriever_name, color=colors[idx]
            )
        
        ax.set_xlabel('K', fontsize=14)
        ax.set_ylabel('nDCG@K', fontsize=14)
        ax.set_title('nDCG at Different K Values', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        filepath = self.output_dir / save_name
        plt.savefig(filepath)
        print(f"Saved plot: {filepath}")
        plt.close()
    
    def generate_all_comparison_plots(
        self,
        results: Dict,
        k_values: List[int] = [1, 3, 5, 10]
    ):
        """
        Generate all comparison plots for a benchmark result.
        
        Args:
            results: Comparison results dictionary
            k_values: List of K values for plots
        """
        dataset_name = results.get('dataset', 'unknown')
        prefix = f"{dataset_name}_"
        
        print(f"\nGenerating all plots for dataset: {dataset_name}")
        print("-" * 70)
        
        # Individual metric comparisons
        key_metrics = ['MAP', 'MRR', 'P@10', 'R@10', 'nDCG@10']
        for metric in key_metrics:
            try:
                self.plot_metric_comparison(
                    results, metric,
                    save_name=f"{prefix}{metric.replace('@', '_at_')}.png"
                )
            except Exception as e:
                print(f"Error plotting {metric}: {e}")
        
        # All metrics comparison
        try:
            self.plot_all_metrics_comparison(
                results,
                save_name=f"{prefix}all_metrics.png"
            )
        except Exception as e:
            print(f"Error plotting all metrics: {e}")
        
        # Precision-recall curve
        try:
            self.plot_precision_recall_curve(
                results, k_values,
                save_name=f"{prefix}precision_recall.png"
            )
        except Exception as e:
            print(f"Error plotting precision-recall: {e}")
        
        # nDCG at K
        try:
            self.plot_ndcg_at_k(
                results, k_values,
                save_name=f"{prefix}ndcg_at_k.png"
            )
        except Exception as e:
            print(f"Error plotting nDCG@K: {e}")
        
        # Latency comparison
        try:
            self.plot_latency_comparison(
                results,
                save_name=f"{prefix}latency.png"
            )
        except Exception as e:
            print(f"Error plotting latency: {e}")
        
        print(f"\nAll plots generated in: {self.output_dir}")


def load_and_visualize_results(results_dir: str = "experiments/results"):
    """
    Load all result files and generate visualizations.
    
    Args:
        results_dir: Directory containing result JSON files
    """
    results_path = Path(results_dir)
    visualizer = ExperimentVisualizer()
    
    # Find all comparison result files
    comparison_files = list(results_path.glob("comparison_*.json"))
    
    if not comparison_files:
        print(f"No comparison result files found in {results_dir}")
        return
    
    print(f"Found {len(comparison_files)} comparison result files")
    
    for filepath in comparison_files:
        print(f"\nProcessing: {filepath.name}")
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        visualizer.generate_all_comparison_plots(results)