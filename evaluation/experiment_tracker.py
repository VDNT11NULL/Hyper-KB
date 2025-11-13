"""
Experiment tracking and versioning system.
Maintains experiment history and enables reproducible research.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    experiment_id: str
    experiment_name: str
    description: str
    retriever_type: str
    retriever_params: Dict
    dataset_name: str
    dataset_params: Dict
    evaluation_metrics: List[str]
    timestamp: str
    git_commit: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Results from a completed experiment."""
    experiment_id: str
    config: ExperimentConfig
    metrics: Dict[str, float]
    latency_stats: Optional[Dict] = None
    additional_data: Dict = field(default_factory=dict)
    completed_at: Optional[str] = None


class ExperimentTracker:
    """
    Track and version experiments for reproducibility.
    Maintains experiment history and enables comparison.
    """
    
    def __init__(self, experiments_dir: str = "experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiments_dir: Root directory for experiments
        """
        self.experiments_dir = Path(experiments_dir)
        self.configs_dir = self.experiments_dir / "configs"
        self.results_dir = self.experiments_dir / "results"
        self.logs_dir = self.experiments_dir / "logs"
        
        # Create directories
        for directory in [self.configs_dir, self.results_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.experiments_dir / "experiment_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load experiment registry."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {'experiments': []}
    
    def _save_registry(self):
        """Save experiment registry."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _generate_experiment_id(self, config: Dict) -> str:
        """Generate unique experiment ID from configuration."""
        config_str = json.dumps(config, sort_keys=True)
        hash_obj = hashlib.md5(config_str.encode())
        return hash_obj.hexdigest()[:12]
    
    def create_experiment(
        self,
        experiment_name: str,
        description: str,
        retriever_type: str,
        retriever_params: Dict,
        dataset_name: str,
        dataset_params: Dict = None,
        evaluation_metrics: List[str] = None,
        tags: List[str] = None,
        notes: Optional[str] = None
    ) -> ExperimentConfig:
        """
        Create and register a new experiment.
        
        Args:
            experiment_name: Human-readable experiment name
            description: Experiment description
            retriever_type: Type of retriever (sparse/dense/hybrid)
            retriever_params: Retriever configuration parameters
            dataset_name: Name of benchmark dataset
            dataset_params: Dataset configuration parameters
            evaluation_metrics: List of metrics to evaluate
            tags: Tags for categorizing experiment
            notes: Additional notes
            
        Returns:
            ExperimentConfig object
        """
        if dataset_params is None:
            dataset_params = {}
        
        if evaluation_metrics is None:
            evaluation_metrics = ['MAP', 'MRR', 'P@10', 'R@10', 'nDCG@10']
        
        if tags is None:
            tags = []
        
        # Generate experiment ID
        config_dict = {
            'experiment_name': experiment_name,
            'retriever_type': retriever_type,
            'retriever_params': retriever_params,
            'dataset_name': dataset_name,
            'dataset_params': dataset_params
        }
        experiment_id = self._generate_experiment_id(config_dict)
        
        # Create config
        config = ExperimentConfig(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            description=description,
            retriever_type=retriever_type,
            retriever_params=retriever_params,
            dataset_name=dataset_name,
            dataset_params=dataset_params,
            evaluation_metrics=evaluation_metrics,
            timestamp=datetime.now().isoformat(),
            notes=notes,
            tags=tags
        )
        
        # Save config
        config_file = self.configs_dir / f"{experiment_id}.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        # Register experiment
        self.registry['experiments'].append({
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'timestamp': config.timestamp,
            'tags': tags
        })
        self._save_registry()
        
        print(f"Created experiment: {experiment_name}")
        print(f"Experiment ID: {experiment_id}")
        
        return config
    
    def log_result(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        latency_stats: Optional[Dict] = None,
        additional_data: Optional[Dict] = None
    ):
        """
        Log experiment results.
        
        Args:
            experiment_id: Experiment identifier
            metrics: Dictionary of metric values
            latency_stats: Latency statistics
            additional_data: Additional data to store
        """
        # Load config
        config_file = self.configs_dir / f"{experiment_id}.json"
        if not config_file.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        config = ExperimentConfig(**config_dict)
        
        # Create result
        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            metrics=metrics,
            latency_stats=latency_stats,
            additional_data=additional_data or {},
            completed_at=datetime.now().isoformat()
        )
        
        # Save result
        result_file = self.results_dir / f"{experiment_id}.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        print(f"Logged results for experiment: {experiment_id}")
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration by ID."""
        config_file = self.configs_dir / f"{experiment_id}.json"
        if not config_file.exists():
            return None
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        return ExperimentConfig(**config_dict)
    
    def get_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get experiment result by ID."""
        result_file = self.results_dir / f"{experiment_id}.json"
        if not result_file.exists():
            return None
        
        with open(result_file, 'r') as f:
            result_dict = json.load(f)
        
        # Reconstruct nested objects
        result_dict['config'] = ExperimentConfig(**result_dict['config'])
        
        return ExperimentResult(**result_dict)
    
    def list_experiments(
        self,
        tags: Optional[List[str]] = None,
        retriever_type: Optional[str] = None
    ) -> List[Dict]:
        """
        List experiments with optional filtering.
        
        Args:
            tags: Filter by tags
            retriever_type: Filter by retriever type
            
        Returns:
            List of experiment summaries
        """
        experiments = self.registry['experiments']
        
        if tags:
            experiments = [
                exp for exp in experiments
                if any(tag in exp.get('tags', []) for tag in tags)
            ]
        
        if retriever_type:
            filtered = []
            for exp in experiments:
                config = self.get_experiment(exp['experiment_id'])
                if config and config.retriever_type == retriever_type:
                    filtered.append(exp)
            experiments = filtered
        
        return experiments
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        Compare results across multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Specific metrics to compare (if None, uses all)
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            'experiment_ids': experiment_ids,
            'experiments': [],
            'metric_comparison': {}
        }
        
        for exp_id in experiment_ids:
            result = self.get_result(exp_id)
            if result:
                comparison['experiments'].append({
                    'experiment_id': exp_id,
                    'name': result.config.experiment_name,
                    'retriever_type': result.config.retriever_type,
                    'metrics': result.metrics
                })
        
        # Build metric comparison
        if comparison['experiments']:
            all_metrics = set()
            for exp in comparison['experiments']:
                all_metrics.update(exp['metrics'].keys())
            
            if metrics:
                all_metrics = all_metrics.intersection(set(metrics))
            
            for metric in all_metrics:
                comparison['metric_comparison'][metric] = {
                    exp['name']: exp['metrics'].get(metric, 0.0)
                    for exp in comparison['experiments']
                }
        
        return comparison
    
    def generate_comparison_report(
        self,
        experiment_ids: List[str],
        output_file: Optional[str] = None
    ):
        """
        Generate markdown comparison report.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            output_file: Output markdown file path
        """
        comparison = self.compare_experiments(experiment_ids)
        
        report_lines = [
            "# Experiment Comparison Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## Experiments Compared: {len(experiment_ids)}\n"
        ]
        
        # Experiment details
        for exp in comparison['experiments']:
            report_lines.append(f"### {exp['name']}")
            report_lines.append(f"- **Experiment ID**: {exp['experiment_id']}")
            report_lines.append(f"- **Retriever Type**: {exp['retriever_type']}\n")
        
        # Metric comparison table
        report_lines.append("## Metric Comparison\n")
        report_lines.append("| Metric | " + " | ".join(
            exp['name'] for exp in comparison['experiments']
        ) + " |")
        report_lines.append("|" + "---|" * (len(comparison['experiments']) + 1))
        
        for metric, values in comparison['metric_comparison'].items():
            row = f"| {metric} |"
            for exp in comparison['experiments']:
                value = values.get(exp['name'], 0.0)
                row += f" {value:.4f} |"
            report_lines.append(row)
        
        report = "\n".join(report_lines)
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Comparison report saved to: {output_path}")
        else:
            print(report)
        
        return report