# Hyper-KB: Conversational Information Retrieval System

![System Architecture](pipeline/Hyper-KB-Pipeline-Arch-v4.png)

A sophisticated hybrid information retrieval system designed for conversational AI applications, featuring LLM-powered content curation, multi-strategy retrieval, and intelligent drift detection for maintaining contextual coherence across extended conversations.

## Overview

Hyper-KB addresses the fundamental challenge of maintaining conversational context in multi-turn interactions by combining sparse lexical retrieval (BM25), dense semantic search (FAISS), and hybrid fusion strategies with real-time topic drift detection. The system automatically curates incoming interactions, extracts structured metadata, and adapts retrieval strategies based on conversational dynamics.

### Core Capabilities

**Intelligent Curation Pipeline**
- Automated keyword and entity extraction using HuggingFace's gemma-3-27b-it model
- Structured metadata generation for enhanced retrieval
- Embedding generation with sentence-transformers for semantic search
- MongoDB-based persistent storage with rich metadata tracking

**Multi-Strategy Retrieval**
- BM25 sparse retrieval for lexical matching
- FAISS dense retrieval for semantic similarity
- Hybrid fusion with three methods: Reciprocal Rank Fusion (RRF), weighted linear combination, and distribution-based normalization
- Session-aware filtering and context-based re-ranking

**Drift-Aware Context Management**
- Real-time topic shift detection between conversation turns
- Conversational state tracking across multiple sessions
- Adaptive retrieval strategies that adjust based on detected drift
- Context stability scoring for maintaining coherence

**Prompt Enhancement Framework**
- Multiple aggregation strategies: concatenation, template-based, weighted, and recency-aware
- Configurable prompt templates for different use cases
- Token management and context truncation
- Metadata integration for drift-aware prompting

## System Architecture

The system follows a modular pipeline architecture:

1. **Ingestion Layer**: LLM-based curator processes query-response pairs, extracting features and generating embeddings
2. **Storage Layer**: MongoDB stores interactions with rich metadata including temporal markers, drift scores, and retrieval history
3. **Retrieval Layer**: Parallel execution of sparse and dense retrieval followed by hybrid fusion
4. **Drift Detection Layer**: Semantic similarity analysis between consecutive turns with configurable thresholds
5. **Context Management Layer**: Session-aware tracking with adaptive retrieval bias calculation
6. **Enhancement Layer**: Context aggregation and prompt construction with template management

## Installation

### Prerequisites

- Python 3.8 or higher
- MongoDB Atlas account or local MongoDB instance
- HuggingFace API token for LLM inference

### Setup

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd Hyper-KB
pip install -r requirements.txt
```

Configure environment variables:

```bash
export HF_TOKEN="your_huggingface_token"
```

Update MongoDB connection string in `kb/kb_enhanced.py` if using custom configuration.

### Verification

Run the quick start script to verify installation:

```bash
python scripts/quick_start.py
```

## Quick Start

### Basic Usage

```python
from pipeline import HybridRetrievalPipeline
from uuid import uuid4

# Initialize pipeline
pipeline = HybridRetrievalPipeline(
    db_name="my_knowledge_base",
    fusion_method='rrf',
    aggregation_strategy='weighted'
)

# Create session
session_id = str(uuid4())

# Store interactions
result = pipeline.process_interaction(
    query="What is machine learning?",
    response="Machine learning is a subset of AI that enables systems to learn from data.",
    session_id=session_id
)

# Query with context enhancement
query_result = pipeline.query(
    query="Explain neural networks",
    session_id=session_id,
    top_k=5
)

# Access enhanced prompt
enhanced_prompt = query_result['enhanced_prompt']
```

### Advanced Configuration

```python
# Configure with custom parameters
pipeline = HybridRetrievalPipeline(
    db_name="advanced_kb",
    fusion_method='weighted',  # Options: 'rrf', 'weighted', 'distribution'
    aggregation_strategy='recency',  # Options: 'concatenate', 'weighted', 'recency'
    prompt_template='research'  # Options: 'conversational', 'qa', 'research'
)

# Use adaptive retrieval based on drift
result = pipeline.query(
    query="Your question here",
    session_id=session_id,
    top_k=10,
    use_adaptive=True
)
```

## Project Structure

```
Hyper-KB/
├── curator/              # LLM-based curation and metadata extraction
│   ├── curator_module.py
│   ├── orchestrator.py
│   └── llm_client.py
├── kb/                   # Storage backend implementations
│   ├── kb_base.py       # Abstract base classes
│   ├── kb_enhanced.py   # MongoDB implementation
│   └── migration_script.py
├── retrieval/           # Retrieval implementations
│   ├── sparse_retrieval.py
│   ├── dense_retrieval.py
│   └── hybrid_fusion.py
├── drift/               # Drift detection and adaptive retrieval
│   ├── drift_detector.py
│   ├── context_tracker.py
│   └── adaptive_retrieval.py
├── prompt_enhancement/  # Context aggregation and prompt building
│   ├── context_aggregator.py
│   ├── prompt_builder.py
│   └── strategies.py
├── evaluation/          # Benchmarking and metrics
│   ├── metrics.py
│   ├── benchmarks.py
│   ├── visualization.py
│   └── experiment_tracker.py
├── pipeline/            # End-to-end pipeline orchestration
│   ├── end_to_end.py
│   └── e2e_pipeline_test.py
└── scripts/             # Utility and demonstration scripts
    ├── seed_data.py
    ├── run_evaluation.py
    └── quick_start.py
```

## Evaluation and Benchmarking

### Running Evaluations

The system includes comprehensive evaluation tools for measuring retrieval performance:

```bash
# Seed database with sample data
python scripts/seed_data.py

# Run complete evaluation suite
python scripts/run_evaluation.py
```

This generates:
- Retrieval metrics (MAP, MRR, nDCG, Precision, Recall)
- Latency statistics (mean, P95, P99)
- Comparison plots for different retrieval methods
- JSON results in `experiments/results/`

### Supported Metrics

**Information Retrieval Metrics**
- Mean Average Precision (MAP)
- Mean Reciprocal Rank (MRR)
- Precision@K, Recall@K, F1@K
- Normalized Discounted Cumulative Gain (nDCG@K)

**Performance Metrics**
- Query latency distributions
- Throughput (queries per second)
- Index build times

**Drift-Specific Metrics**
- Topic shift detection accuracy
- Context stability scores
- Adaptive retrieval effectiveness

### Experiment Tracking

Track and version experiments using the built-in tracker:

```python
from evaluation.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()

# Create experiment
config = tracker.create_experiment(
    experiment_name="BM25_Baseline",
    description="Baseline sparse retrieval",
    retriever_type="sparse",
    retriever_params={'k1': 1.5, 'b': 0.75},
    dataset_name="conversational_dataset"
)

# Log results
tracker.log_result(
    experiment_id=config.experiment_id,
    metrics=result_metrics,
    latency_stats=latency_stats
)

# Generate comparison report
tracker.generate_comparison_report(
    experiment_ids=['exp1', 'exp2', 'exp3'],
    output_file="experiments/comparison_report.md"
)
```

## Configuration

### Retriever Configuration

Configure retrieval parameters via YAML:

```yaml
retriever:
  type: "hybrid"
  sparse_method: "bm25"
  dense_method: "faiss"
  fusion_method: "rrf"
  
  sparse_params:
    k1: 1.5
    b: 0.75
  
  dense_params:
    dimension: 384
    index_type: "flat"
  
  fusion_params:
    k: 60
```

Load and execute:

```bash
python scripts/load_config_and_run.py experiments/configs/your_config.yaml
```

### Drift Detection Parameters

```python
from drift import DriftDetector

detector = DriftDetector(
    shift_threshold=0.4,  # Similarity threshold for shift detection
    drift_window=3        # Number of previous turns to consider
)
```

## Development and Testing

### Running Tests

Execute the end-to-end test suite:

```bash
python pipeline/e2e_pipeline_test.py
```

The test suite validates:
- Data ingestion and curation pipeline
- Feature extraction and embedding generation
- Sparse, dense, and hybrid retrieval
- Drift detection accuracy
- Context tracking and adaptive retrieval
- Prompt enhancement and aggregation
- Performance and latency benchmarks

Test results are saved to `experiments/results/tests/`.

### Adding Custom Components

**Custom Retrieval Method**

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

**Custom Fusion Strategy**

```python
from retrieval.hybrid_fusion import HybridFusion

class CustomFusion(HybridFusion):
    def fuse(self, sparse_results, dense_results, top_k):
        # Implementation
        pass
```

**Custom Aggregation Strategy**

```python
from prompt_enhancement.strategies import AggregationStrategy

class CustomAggregation(AggregationStrategy):
    def aggregate(self, results, max_tokens):
        # Implementation
        pass
```

## Use Cases

**Research Applications**
- Literature review with conversational context
- Multi-document question answering
- Academic knowledge management

**Enterprise Applications**
- Internal documentation retrieval
- Customer support knowledge bases
- Technical documentation systems

**Development Applications**
- Code example retrieval
- API documentation search
- Development knowledge sharing

## Performance Characteristics

Based on evaluation with synthetic conversational datasets:

- **Retrieval Latency**: Average 200-500ms per query (depending on corpus size)
- **Indexing Time**: Linear with corpus size, approximately 1000 docs/second
- **Memory Usage**: BM25 index ~50MB per 10K documents, FAISS index ~150MB per 10K documents
- **Drift Detection Overhead**: ~20-30ms per turn
- **Context Aggregation**: ~10-15ms per query

<!-- Performance scales well with corpus size up to 100K documents. For larger deployments, consider sharding st/rategies or approximate search methods. -->

## Technical Details

### Retrieval Fusion

**Reciprocal Rank Fusion (RRF)**
```
score(d) = Σ(1 / (k + rank_i))
```
where k=60 (constant from Cormack et al. 2009), summed over all rankings containing document d.

**Weighted Linear Fusion**
```
score(d) = α × sparse_score(d) + (1-α) × dense_score(d)
```
with optional min-max normalization.

**Distribution-Based Fusion**
```
score(d) = α × z_sparse(d) + (1-α) × z_dense(d)
```
using z-score normalization for robustness across different score distributions.

### Drift Detection

Computes cosine similarity between consecutive turn embeddings:
```
similarity(t, t-1) = (e_t · e_{t-1}) / (||e_t|| × ||e_{t-1}||)
shift_detected = similarity < threshold
```

Adaptive retrieval adjusts three parameters based on drift:
- **Recency Weight**: Increases after shifts to favor recent interactions
- **Session Filter Strength**: Decreases after shifts to expand search scope
- **Global Search Weight**: Increases after shifts for broader context

### Context Aggregation

**Weighted Strategy**: Applies relevance-based prefixes based on normalized scores
- Score > 0.8: [HIGHLY RELEVANT]
- Score > 0.5: [RELEVANT]
- Score ≤ 0.5: [POTENTIALLY RELEVANT]

**Recency-Aware Strategy**: Orders by timestamp and includes turn-based markers
```
[N turns ago] Query: ... Answer: ...
```

## Limitations and Considerations

**Computational Requirements**
- LLM inference requires HuggingFace API access or local GPU
- FAISS indexing memory scales linearly with corpus size
- MongoDB storage scales with interaction volume

**Drift Detection Accuracy**
- Threshold tuning required for domain-specific applications
- Performance varies with conversation topic coherence
- May produce false positives in information-seeking dialogues

**Scalability**
- Current implementation optimized for <100K documents
- Large-scale deployments require distributed indexing
- MongoDB sharding recommended for >1M interactions


## References

This implementation builds upon established research in information retrieval and conversational AI:

- Cormack, G. V., et al. (2009). "Reciprocal rank fusion outperforms condorcet and individual rank learning methods"
- Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond"
- Johnson, J., et al. (2019). "Billion-scale similarity search with GPUs" (FAISS)
