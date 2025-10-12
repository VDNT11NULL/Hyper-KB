import os
import json
import numpy as np
import dspy
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from kb import KnowledgeBase

# ----------------------------- #
# Define DSPy Signature
# ----------------------------- #
CuratorSignature = dspy.Signature("query_text, response_text -> keywords, entities, context_passage")

# ----------------------------- #
# CuratorModule (LLM-powered)
# ----------------------------- #
class CuratorModule(dspy.Module):
    def __init__(self, embed_model="all-MiniLM-L6-v2"):
        super().__init__()
        self.signature = CuratorSignature

        # Initialize Hugging Face InferenceClient with proper chat endpoint
        self.client = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            token=os.getenv("HF_TOKEN")
        )

        # Sentence transformer for embeddings
        self.embedder = SentenceTransformer(embed_model)

    def forward(self, query_text: str, response_text: str):
        """
        Generates structured features (keywords, entities, summary)
        using the Hugging Face model.
        """
        # Enhanced prompt with clear entity classification rules and examples
        prompt = f"""[INST] You are a curator that extracts structured information from conversations.

Query: {query_text}
Response: {response_text}

IMPORTANT ENTITY CLASSIFICATION RULES:
- PERSON: Individual people, authors, researchers, scientists (e.g., "John Smith", "Einstein", "Vaswani et al.", "Dr. Jane Doe")
- ORG: Organizations, companies, institutions, universities (e.g., "Google", "MIT", "United Nations", "OpenAI")
- DATE: Years, dates, time periods (e.g., "2017", "March 2020", "19th century")
- MISC: Technologies, models, products, concepts that are NOT people or organizations (e.g., "GPT-3", "iPhone", "transformer model", "CRISPR")

EXAMPLES:
Text: "Einstein developed relativity theory. Google released BERT in 2018."
Entities:
- PERSON: ["Einstein"]
- ORG: ["Google"]
- DATE: ["2018"]
- MISC: ["relativity theory", "BERT"]

Text: "The paper by Vaswani et al. introduced transformers at NeurIPS."
Entities:
- PERSON: ["Vaswani et al."]
- ORG: ["NeurIPS"]
- DATE: []
- MISC: ["transformers"]

Now extract from the given Query and Response.

Return ONLY a valid JSON object:
{{
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "entities": {{
    "PERSON": [],
    "ORG": [],
    "DATE": [],
    "MISC": []
  }},
  "context_passage": "A comprehensive summary of the query and response in 2-3 sentences."
}}

Return ONLY the JSON, nothing else. [/INST]"""

        # Generate text using InferenceClient
        try:
            # Use chat_completion for better results with instruct models
            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=400,  # Increased for better extraction
                temperature=0.2  # Lower temperature for more consistent output
            )
            
            gen_text = response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Error during text generation: {e}")

        print("\n=== RAW MODEL OUTPUT ===\n", gen_text, "\n=========================\n")

        # Safe JSON parsing with better extraction
        parsed = self._parse_json_safely(gen_text)

        # If parsing failed completely, use fallback extraction
        if not parsed or not any(parsed.values()):
            parsed = self._create_fallback_response(query_text, response_text)

        # Post-process entities to fix common misclassifications
        parsed = self._fix_entity_classification(parsed)

        return parsed

    def _fix_entity_classification(self, data: dict) -> dict:
        """Post-process to fix common entity misclassifications"""
        entities = data.get("entities", {})
        
        # Common person indicators
        person_indicators = ["et al.", "dr.", "prof.", "mr.", "ms.", "mrs."]
        
        # Move misclassified persons from MISC to PERSON
        misc_items = entities.get("MISC", [])
        persons = entities.get("PERSON", [])
        
        items_to_move = []
        for item in misc_items:
            item_lower = item.lower()
            # Check if it contains person indicators
            if any(indicator in item_lower for indicator in person_indicators):
                items_to_move.append(item)
                persons.append(item)
        
        # Remove moved items from MISC
        entities["MISC"] = [item for item in misc_items if item not in items_to_move]
        entities["PERSON"] = persons
        
        # Common model/product names that should stay in MISC
        # (GPT-3, GPT-4, Claude, BERT, etc. are correctly in MISC)
        
        data["entities"] = entities
        return data

    def _parse_json_safely(self, text: str) -> dict:
        """Enhanced JSON parsing with multiple strategies"""
        # Strategy 1: Direct parse
        try:
            parsed = json.loads(text)
            if self._validate_structure(parsed):
                return parsed
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON block
        import re
        
        # Try to find JSON between ```json and ``` or just between {}
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{[^{}]*\{[^{}]*\}[^{}]*\})',  # nested braces
            r'(\{[^{}]+\})'  # simple braces
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    if self._validate_structure(parsed):
                        return parsed
                except:
                    continue

        # Strategy 3: Manual key extraction with better entity parsing
        try:
            result = {}
            
            # Extract keywords
            kw_match = re.search(r'"keywords"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if kw_match:
                kw_str = kw_match.group(1)
                keywords = [k.strip(' "\'') for k in kw_str.split(',') if k.strip()]
                result['keywords'] = keywords
            
            # Extract context_passage
            ctx_match = re.search(r'"context_passage"\s*:\s*"([^"]*)"', text, re.DOTALL)
            if ctx_match:
                result['context_passage'] = ctx_match.group(1)
            
            # Extract entities with better handling
            entities = {"PERSON": [], "ORG": [], "DATE": [], "MISC": []}
            
            # Try to extract the entire entities object
            ent_match = re.search(r'"entities"\s*:\s*\{(.*?)\}', text, re.DOTALL)
            if ent_match:
                ent_content = ent_match.group(1)
                
                # Extract each entity type
                for entity_type in ["PERSON", "ORG", "DATE", "MISC"]:
                    type_match = re.search(
                        rf'"{entity_type}"\s*:\s*\[(.*?)\]',
                        ent_content,
                        re.DOTALL
                    )
                    if type_match:
                        items_str = type_match.group(1)
                        items = [
                            item.strip(' "\'') 
                            for item in items_str.split(',') 
                            if item.strip() and item.strip() not in ['""', "''"]
                        ]
                        entities[entity_type] = items
            
            result['entities'] = entities
            
            if result:
                return self._fill_defaults(result)
                
        except Exception as e:
            print(f"⚠️ Manual extraction failed: {e}")

        return {}

    def _validate_structure(self, data: dict) -> bool:
        """Check if parsed data has expected structure"""
        return (
            isinstance(data, dict) and
            'keywords' in data and
            'entities' in data and
            'context_passage' in data
        )

    def _fill_defaults(self, data: dict) -> dict:
        """Fill missing keys with defaults"""
        return {
            "keywords": data.get("keywords", []),
            "entities": data.get("entities", {"PERSON": [], "ORG": [], "DATE": [], "MISC": []}),
            "context_passage": data.get("context_passage", "")
        }

    def _create_fallback_response(self, query: str, response: str) -> dict:
        """Create a basic response when LLM fails"""
        import re
        
        # Simple keyword extraction
        words = (query + " " + response).lower().split()
        keywords = [w.strip('.,!?') for w in words if len(w) > 4][:5]
        
        # Try to extract years as dates
        dates = re.findall(r'\b(19\d{2}|20\d{2})\b', query + " " + response)
        
        return {
            "keywords": list(set(keywords)),
            "entities": {
                "PERSON": [], 
                "ORG": [], 
                "DATE": list(set(dates)), 
                "MISC": []
            },
            "context_passage": f"Discussion about: {query[:100]}"
        }

# ----------------------------- #
# CuratorLLM Orchestrator
# ----------------------------- #
class CuratorLLM:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.curator = CuratorModule()

    def curate_and_store(self, query_text: str, response_text: str, session_id: str):
        """
        Full pipeline: generate features → embedding → store in KB.
        """
        output = self.curator(query_text=query_text, response_text=response_text)

        # Create embedding from the generated summary (or fallback)
        text_to_embed = (
            output.get("context_passage")
            or (query_text + " " + response_text)
        )
        embedding_vector = self.curator.embedder.encode(text_to_embed)

        # Print everything before inserting
        # print("\n ===== Curator LLM Extracted Data =====")
        # print(f"Query: {query_text}")
        # print(f"Response: {response_text}\n")
        # print(f" Keywords: {output.get('keywords', [])}")
        # print(f" Entities: {json.dumps(output.get('entities', {}), indent=2, ensure_ascii=False)}")
        # print(f"Context (Summary): {output.get('context_passage', '')}\n")
        # print(f"Embedding vector length: {len(embedding_vector)}")
        # print("========================================\n")

        interaction_id = self.kb.insert_interaction(
            query_text=query_text,
            response_text=response_text,
            session_id=session_id,
            keywords=output.get("keywords", []),
            entities=output.get("entities", {}),
            context_passage=output.get("context_passage", ""),
            embedding_vector=np.array(embedding_vector, dtype=float)
        )

        return interaction_id

# ----------------------------- #
# Test Run
# ----------------------------- #
if __name__ == "__main__":
    from uuid import uuid4

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

    iid = curator_llm.curate_and_store(
        query_text=query_1,
        response_text=response_1,
        session_id=sid
    )

    print(f"Interaction inserted: {iid}")
    print("Stats:", kb.get_stats())
    kb.close()