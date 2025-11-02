import json
import re
import dspy
from .signature import CuratorSignature
from .llm_client import get_llm_client, get_embedder
from .parser_utils import (
    parse_json_safely,
    fix_entity_classification,
    validate_structure,
    fill_defaults
)

class CuratorModule(dspy.Module):
    def __init__(self, embed_model="all-MiniLM-L6-v2"):
        super().__init__()
        self.signature = CuratorSignature
        self.client = get_llm_client()
        self.embedder = get_embedder(embed_model)

    def forward(self, query_text: str, response_text: str):
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
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat_completion(messages=messages, max_tokens=500, temperature=0.2)
            gen_text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error during text generation: {e}")
            gen_text = ""

        print("\n=== RAW MODEL OUTPUT ===\n", gen_text, "\n=========================\n")

        parsed = parse_json_safely(gen_text, validate_structure, fill_defaults)
        if not parsed or not any(parsed.values()):
            parsed = self._create_fallback_response(query_text, response_text)
        parsed = fix_entity_classification(parsed)
        return parsed

    def _create_fallback_response(self, query, response):
        import re
        words = (query + " " + response).lower().split()
        keywords = [w.strip('.,!?') for w in words if len(w) > 4][:5]
        dates = re.findall(r'\b(19\d{2}|20\d{2})\b', query + " " + response)
        return {
            "keywords": list(set(keywords)),
            "entities": {"PERSON": [], "ORG": [], "DATE": list(set(dates)), "MISC": []},
            "context_passage": f"Discussion about: {query[:100]}"
        }
