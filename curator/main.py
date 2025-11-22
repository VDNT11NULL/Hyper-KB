from uuid import uuid4
from kb import KnowledgeBase
from .orchestrator import CuratorLLM

if __name__ == "__main__":
    kb = KnowledgeBase()
    curator_llm = CuratorLLM(kb)
    sid = str(uuid4())

    query_1 = """Can you explain the difference between transformer architecture and traditional 
    RNN/LSTM models? Also, why have transformers become so dominant in NLP tasks?
    """

    response_1 = """Great question! The key differences between transformers and RNN/LSTM models are fundamental 
    to understanding modern NLP.

    Traditional RNNs and LSTMs process sequences sequentially - they read text word by word, 
    maintaining a hidden state that gets updated at each step. This sequential nature creates 
    two major problems: first, they struggle with long-range dependencies because information 
    from early in the sequence can get lost or diluted as it propagates through many time steps. 
    Second, this sequential processing prevents parallelization during training, making them 
    computationally inefficient.

    Transformers, introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al., 
    revolutionized this approach through the self-attention mechanism. Instead of processing 
    sequentially, transformers can attend to all positions in the input simultaneously. The 
    self-attention mechanism computes attention scores between every pair of tokens, allowing 
    the model to directly capture relationships regardless of distance in the sequence.

    The architecture consists of encoder and decoder blocks. Each encoder block has two main 
    components: a multi-head self-attention layer and a feed-forward neural network. Multi-head 
    attention allows the model to focus on different aspects of the input simultaneously - some 
    heads might focus on syntax, others on semantics, and others on long-range dependencies.

    Transformers became dominant for several reasons. First, they're highly parallelizable since 
    all tokens are processed simultaneously, enabling efficient training on GPUs. Second, they 
    handle long-range dependencies much better through direct attention connections. Third, they 
    scale remarkably well - larger transformers with more parameters consistently perform better, 
    leading to models like GPT-3, GPT-4, and Claude. Fourth, positional encodings allow them to 
    maintain sequence order information without sequential processing.

    The attention mechanism's interpretability is also valuable - we can visualize what the model 
    focuses on, providing insights into its decision-making process. This has led to transformers 
    dominating not just NLP, but also computer vision (Vision Transformers), protein folding 
    (AlphaFold), and multimodal tasks.
    """
    iid = curator_llm.curate_and_store(query_1, response_1, sid)
    print(f"Interaction inserted: {iid}")
    print("Stats:", kb.get_stats())
    kb.close()
