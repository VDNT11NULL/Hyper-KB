
# Hybrid Information Retrieval with LLM-Curated Knowledge Bases

A research project implementing hybrid retrieval systems combining sparse (BM25, FTS) and dense (FAISS) methods with LLM-based knowledge curation for conversational AI applications.

## Project Overview

This project implements a complete hybrid retrieval pipeline with:

- **LLM-Based Curation**: Automatic extraction of keywords, entities, and context passages
- **Dual Retrieval Methods**: Sparse (lexical) and dense (semantic) retrieval
- **Multiple Fusion Strategies**: RRF, weighted linear, distribution-based
- **Conversational Drift Handling**: Topic shift detection and context stability tracking
- **Comprehensive Evaluation**: Standard IR metrics, latency analysis, ablation studies

## Directory Structure

```
hybrid_retrieval_project/
├── kb/                          # Knowledge Base module
│   ├── kb_base.py              # Abstract base classes
│   ├── kb_enhanced.py          # Enhanced MongoDB implementation
│   └── migration_script.py     # Data migration utilities
├── retrieval/                   # Retrieval implementations
│   ├── sparse_retrieval.py     # BM25 and MongoDB FTS
│   ├── dense_retrieval.py      # FAISS-based dense retrieval
│   └── hybrid_fusion.py        # Fusion strategies
├── curator/                     # Curation module
│   ├── curator_module.py       # DSPy-based curator
│   ├── orchestrator.py         # Integration orchestrator
│   └── llm_client.py          # HuggingFace client
├── evaluation/                  # Evaluation and benchmarking
│   ├── metrics.py              # IR metrics (MAP, nDCG, etc.)
│   ├── benchmarks.py           # Benchmark datasets
│   ├── visualization.py        # Plotting utilities
│   └── experiment_tracker.py   # Experiment versioning
├── experiments/                 # Experiment artifacts
│   ├── configs/                # Experiment configurations
│   ├── results/                # Results and plots
│   └── logs/                   # Experiment logs
├── scripts/                     # Utility scripts
│   ├── seed_data.py           # Sample data generation
│   └── run_experiments.py      # Complete experiment runner
└── notebooks/                   # Jupyter notebooks
    └── 01_quickstart.ipynb     # Interactive demo
```

## Installation

### Prerequisites

- Python 3.8+
- MongoDB Atlas account (or local MongoDB)
- HuggingFace account (for API access)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd hybrid_retrieval_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export HF_TOKEN="your_huggingface_token"
```

4. Update MongoDB connection string in `kb/kb_enhanced.py` if needed.

## Quick Start

### 1. Interactive Notebook

Launch Jupyter and open `notebooks/01_quickstart.ipynb` for an interactive demonstration.

```bash
jupyter notebook notebooks/01_quickstart.ipynb
```

### 2. Command Line Usage

```python
from kb.kb_enhanced import EnhancedKnowledgeBase
from retrieval.sparse_retrieval import BM25Retriever
from retrieval.dense_retrieval import FAISSRetriever
from retrieval.hybrid_fusion import HybridRetriever
from curator.orchestrator import CuratorLLM

# Initialize system
kb = EnhancedKnowledgeBase()
curator_llm = CuratorLLM(kb)

# Add interaction
interaction_id = curator_llm.curate_and_store(
    query_text="What is machine learning?",
    response_text="Machine learning is...",
    session_id="session_123"
)

# Initialize retrievers
sparse = BM25Retriever(kb)
sparse.index_documents()

dense = FAISSRetriever(kb, dimension=384)
dense.index_embeddings()

hybrid = HybridRetriever(sparse, dense, fusion_method='rrf')

# Retrieve
query_embedding = curator_llm.get_embedding_for_query("machine learning")
results = hybrid.search(
    query="machine learning",
    query_embedding=query_embedding,
    top_k=5
)
```

### 3. Run Complete Experiment Suite

```bash
python scripts/run_experiments.py
```

This will:
- Seed database with sample conversations
- Initialize all retriever variants
- Run comprehensive benchmarks
- Generate comparison plots
- Create experiment reports

Results saved to: `experiments/results/`

## Key Features

### 1. Knowledge Base

**EnhancedKnowledgeBase** with rich metadata:
- Temporal tracking (timestamps, turn numbers)
- Conversational context (session IDs, dialogue acts)
- Retrieval metadata (access counts, relevance scores)
- Drift indicators (topic shift scores, context stability)

### 2. Retrieval Methods

**Sparse Retrievers:**
- BM25Retriever: Tunable parameters (k1, b)
- MongoFTSRetriever: Native MongoDB full-text search

**Dense Retrievers:**
- FAISSRetriever: Exact and approximate similarity search
- Support for flat and IVF indexes

**Hybrid Fusion:**
- ReciprocalRankFusion (RRF): Parameter-free fusion
- WeightedFusion: Tunable linear combination
- DistributionBasedFusion: Z-score normalization

### 3. Evaluation Metrics

Standard IR metrics:
- Mean Average Precision (MAP)
- Mean Reciprocal Rank (MRR)
- Precision@K, Recall@K, F1@K
- Normalized Discounted Cumulative Gain (nDCG@K)

Performance metrics:
- Latency statistics (mean, median, P95, P99)
- Throughput (queries per second)

Drift metrics:
- Context stability scores
- Topic shift detection accuracy

### 4. Experiment Tracking

**ExperimentTracker** provides:
- Unique experiment IDs
- Configuration versioning
- Result storage
- Comparison utilities
- Markdown report generation

### 5. Visualization

Automated generation of publication-ready plots:
- Metric comparison bar charts
- Precision-recall curves
- nDCG@K line plots
- Latency comparison charts
- All saved in high resolution (300 DPI)

## Evaluation Workflow

1. **Setup Experiment**
```python
from evaluation.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()
config = tracker.create_experiment(
    experiment_name="BM25_Baseline",
    description="BM25 with default parameters",
    retriever_type="sparse",
    retriever_params={'k1': 1.5, 'b': 0.75},
    dataset_name="synthetic_conversational",
    tags=['sparse', 'baseline']
)
```

2. **Run Benchmark**
```python
from evaluation.benchmarks import BenchmarkRunner

runner = BenchmarkRunner()
result = runner.run_retrieval_benchmark(
    retriever=bm25_retriever,
    dataset=benchmark_dataset,
    retriever_name="BM25"
)
```

3. **Log Results**
```python
tracker.log_result(
    experiment_id=config.experiment_id,
    metrics=result['metrics'],
    latency_stats=result['latency']
)
```

4. **Compare Experiments**
```python
comparison = tracker.compare_experiments(
    experiment_ids=['exp1', 'exp2', 'exp3']
)
```

5. **Generate Visualizations**
```python
from evaluation.visualization import ExperimentVisualizer

visualizer = ExperimentVisualizer()
visualizer.generate_all_comparison_plots(comparison)
```

## Benchmarking

### Synthetic Dataset Generation

```python
from evaluation.benchmarks import SyntheticBenchmarkGenerator

dataset = SyntheticBenchmarkGenerator.generate_conversational_dataset(
    num_sessions=10,
    turns_per_session=(3, 8),
    topic_shift_probability=0.2
)
```

### Custom Benchmarks

Define custom benchmark queries:

```python
from evaluation.benchmarks import BenchmarkQuery, BenchmarkDataset

query = BenchmarkQuery(
    query_id="q1",
    query_text="What is supervised learning?",
    relevant_docs={'doc1', 'doc2', 'doc3'},
    relevance_scores={'doc1': 1.0, 'doc2': 0.8, 'doc3': 0.5},
    session_id="session_1",
    turn_number=0
)

dataset = BenchmarkDataset(
    name="custom_benchmark",
    description="Custom ML questions",
    queries=[query]
)
```

## Conversational Drift Handling

The system tracks and handles conversational drift through:

1. **Topic Shift Scores**: Measure semantic distance between consecutive turns
2. **Context Stability**: Track consistency of retrieved contexts
3. **Session-Aware Retrieval**: Filter results by session for context coherence
4. **Turn Number Weighting**: Adjust relevance based on recency in conversation

Example:

```python
interaction_id = curator_llm.curate_and_store(
    query_text="Actually, tell me about reinforcement learning instead",
    response_text="Reinforcement learning...",
    session_id=session_id,
    previous_interaction_id=prev_id,
    topic_shift_score=0.8  # High shift from previous topic
)
```

## Extending the System

### Adding New Retrieval Methods

1. Inherit from `SparseRetriever` or `DenseRetriever`:

```python
from kb.kb_base import SparseRetriever

class CustomRetriever(SparseRetriever):
    def index_documents(self, documents):
        # Implementation
        pass
    
    def search(self, query, top_k, filters):
        # Implementation
        pass
```

2. Register with experiment tracker and benchmark.

### Adding New Fusion Strategies

1. Inherit from `HybridFusion`:

```python
from retrieval.hybrid_fusion import HybridFusion

class CustomFusion(HybridFusion):
    def fuse(self, sparse_results, dense_results, top_k):
        # Implementation
        pass
```

### Adding Custom Metrics

Add to `evaluation/metrics.py`:

```python
@staticmethod
def custom_metric(retrieved, relevant):
    # Implementation
    return score
```

## Results and Reports

After running experiments:

- **Results JSON**: `experiments/results/*.json`
- **Plots**: `experiments/results/plots/*.png`
- **Comparison Reports**: `experiments/results/comparison_report.md`
- **Experiment Registry**: `experiments/experiment_registry.json`

## Research Goals

This project addresses:

1. **Hybrid Retrieval Effectiveness**: Comparing sparse, dense, and hybrid approaches
2. **Fusion Strategy Optimization**: Evaluating different fusion methods
3. **Conversational Drift Handling**: Maintaining context coherence across topic shifts
4. **Scalability**: Performance analysis with growing knowledge bases
5. **Prompt Enhancement**: Using retrieved context to improve LLM responses

## Future Work

- Advanced re-ranking models
- Learned fusion strategies
- Real-time drift detection
- Multi-modal retrieval
- Production deployment optimizations
