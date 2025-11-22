import numpy as np
from .curator_module import CuratorModule
from kb import KnowledgeBase

class CuratorLLM:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.curator = CuratorModule()

    def curate_and_store(self, query_text: str, response_text: str, session_id: str):
        output = self.curator(query_text=query_text, response_text=response_text)
        text_to_embed = output.get("context_passage") or (query_text + " " + response_text)
        embedding_vector = self.curator.embedder.encode(text_to_embed)

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
