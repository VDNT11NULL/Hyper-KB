"""
Prompt building and template management.
"""
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Prompt template structure."""
    system_prompt: str
    context_template: str
    query_template: str
    separator: str = "\n\n"


class PromptBuilder:
    """Builds enhanced prompts with retrieved context."""
    
    DEFAULT_TEMPLATES = {
        'conversational': PromptTemplate(
            system_prompt="You are a helpful AI assistant. Use the provided context from previous conversations.",
            context_template="# Previous Context\n{context}",
            query_template="# Current Query\n{query}"
        ),
        'qa': PromptTemplate(
            system_prompt="Answer the question using the provided context.",
            context_template="Context:\n{context}",
            query_template="Question: {query}\nAnswer:"
        )
    }
    
    def __init__(self, template_name: str = 'conversational'):
        self.template = self.DEFAULT_TEMPLATES.get(template_name, self.DEFAULT_TEMPLATES['conversational'])
    
    def build(self, query: str, context: str, metadata: Optional[Dict] = None) -> str:
        """Build complete prompt."""
        parts = [self.template.system_prompt]
        
        if context:
            parts.append(self.template.context_template.format(context=context))
        
        parts.append(self.template.query_template.format(query=query))
        
        return self.template.separator.join(parts)