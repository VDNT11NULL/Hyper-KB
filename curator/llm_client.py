import os
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

def get_llm_client():
    """Initialize and return a Hugging Face InferenceClient"""
    return InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        token=os.getenv("HF_TOKEN")
    )

def get_embedder(model_name="all-MiniLM-L6-v2"):
    """Return a sentence transformer embedder"""
    return SentenceTransformer(model_name)
