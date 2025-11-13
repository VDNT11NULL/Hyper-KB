"""
Quick start script for immediate use.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import HybridRetrievalPipeline
from uuid import uuid4


def main():
    print("Quick Start - Hybrid Retrieval System")
    print("=" * 70)
    
    # Initialize
    pipeline = HybridRetrievalPipeline()
    session_id = str(uuid4())
    
    # Seed with quick examples
    examples = [
        ("What is Python?", "Python is a high-level programming language."),
        ("How do I write a function?", "Use the def keyword followed by function name and parameters."),
        ("What about cooking?", "Cooking involves preparing food using heat.")
    ]
    
    print("\nStoring examples...")
    for q, r in examples:
        pipeline.process_interaction(q, r, session_id)
    
    # Test query
    print("\nTesting retrieval...")
    result = pipeline.query("How do I cook pasta?", session_id)
    
    print(f"\nEnhanced Prompt:\n{'-'*70}")
    print(result['enhanced_prompt'])
    
    pipeline.close()
    print("\nQuick start complete!")


if __name__ == "__main__":
    main()