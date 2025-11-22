"""
Enhanced prompt building and template management.
Provides flexible, extensible prompt construction with metadata awareness.
"""
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PromptTemplate:
    """
    Prompt template structure with optional metadata integration.
    
    Attributes:
        system_prompt: Initial system-level instruction
        context_template: Template for formatting retrieved context
        query_template: Template for formatting user query
        separator: String separator between prompt sections
        metadata_template: Optional template for metadata integration
        max_context_length: Maximum characters for context section
    """
    system_prompt: str
    context_template: str
    query_template: str
    separator: str = "\n\n"
    metadata_template: Optional[str] = None
    max_context_length: Optional[int] = None
    
    def format_context(self, context: str, **kwargs) -> str:
        """Format context with optional truncation."""
        if self.max_context_length and len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "\n[... context truncated ...]"
        return self.context_template.format(context=context, **kwargs)
    
    def format_query(self, query: str, **kwargs) -> str:
        """Format query with additional parameters."""
        return self.query_template.format(query=query, **kwargs)
    
    def format_metadata(self, metadata: Dict, **kwargs) -> str:
        """Format metadata if template is provided."""
        if self.metadata_template:
            return self.metadata_template.format(**metadata, **kwargs)
        return ""


class PromptBuilder:
    """
    Enhanced prompt builder with support for:
    - Multiple template types
    - Dynamic metadata integration
    - Context truncation and validation
    - Custom template registration
    - Prompt preprocessing and postprocessing
    """
    
    DEFAULT_TEMPLATES = {
        'conversational': PromptTemplate(
            system_prompt="You are a helpful AI assistant. Use the provided context from previous conversations to give accurate and relevant responses.",
            context_template="# Previous Context\n{context}",
            query_template="# Current Query\n{query}",
            metadata_template="[Session: {session_id} | Turn: {turn_number}]",
            max_context_length=4000
        ),
        'qa': PromptTemplate(
            system_prompt="Answer the question using the provided context. If the context doesn't contain enough information, acknowledge this limitation.",
            context_template="Context:\n{context}",
            query_template="Question: {query}\nAnswer:",
            max_context_length=3000
        ),
        'research': PromptTemplate(
            system_prompt="You are a research assistant. Synthesize information from the provided context to give comprehensive, well-sourced answers.",
            context_template="# Research Context\n{context}",
            query_template="# Research Question\n{query}\n\n# Analysis:",
            metadata_template="Sources retrieved: {retrieved_contexts} | Relevance threshold: {relevance_threshold:.2f}",
            max_context_length=5000
        ),
        'code_assistant': PromptTemplate(
            system_prompt="You are a coding assistant. Use the provided context from previous code examples and discussions to help solve programming problems.",
            context_template="# Previous Code Examples\n{context}",
            query_template="# Current Programming Task\n{query}\n\n# Solution:",
            max_context_length=6000
        ),
        'technical_writer': PromptTemplate(
            system_prompt="You are a technical documentation specialist. Use the context to create clear, accurate technical documentation.",
            context_template="# Reference Material\n{context}",
            query_template="# Documentation Request\n{query}\n\n# Documentation:",
            max_context_length=4500
        ),
        'drift_aware': PromptTemplate(
            system_prompt="You are an AI assistant with context awareness. Pay attention to topic shifts and conversation flow.",
            context_template="# Conversation History\n{context}",
            query_template="# Current Query\n{query}",
            metadata_template="⚠️ Topic shift detected: {shift_detected} | Drift score: {topic_shift_score:.2f}",
            max_context_length=4000
        )
    }
    
    def __init__(
        self,
        template_name: str = 'conversational',
        custom_template: Optional[PromptTemplate] = None,
        preprocessors: Optional[List[Callable]] = None,
        postprocessors: Optional[List[Callable]] = None
    ):
        """
        Initialize prompt builder.
        
        Args:
            template_name: Name of predefined template to use
            custom_template: Custom PromptTemplate to use instead
            preprocessors: List of functions to preprocess inputs
            postprocessors: List of functions to postprocess final prompt
        """
        if custom_template:
            self.template = custom_template
        else:
            self.template = self.DEFAULT_TEMPLATES.get(
                template_name,
                self.DEFAULT_TEMPLATES['conversational']
            )
        
        self.template_name = template_name
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []
        self.custom_templates: Dict[str, PromptTemplate] = {}
    
    def register_template(self, name: str, template: PromptTemplate):
        """
        Register a custom template for later use.
        
        Args:
            name: Template identifier
            template: PromptTemplate instance
        """
        self.custom_templates[name] = template
    
    def switch_template(self, template_name: str):
        """
        Switch to a different template.
        
        Args:
            template_name: Name of template (built-in or custom)
        """
        if template_name in self.custom_templates:
            self.template = self.custom_templates[template_name]
        elif template_name in self.DEFAULT_TEMPLATES:
            self.template = self.DEFAULT_TEMPLATES[template_name]
        else:
            raise ValueError(f"Template '{template_name}' not found")
        
        self.template_name = template_name
    
    def _preprocess(self, query: str, context: str, metadata: Optional[Dict]) -> tuple:
        """Apply preprocessing functions."""
        for processor in self.preprocessors:
            query, context, metadata = processor(query, context, metadata)
        return query, context, metadata
    
    def _postprocess(self, prompt: str) -> str:
        """Apply postprocessing functions."""
        for processor in self.postprocessors:
            prompt = processor(prompt)
        return prompt
    
    def build(
        self,
        query: str,
        context: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Build complete prompt from query, context, and metadata.
        
        Args:
            query: User query text
            context: Retrieved context string
            metadata: Optional metadata dictionary
            
        Returns:
            Formatted prompt string
        """
        # Preprocess inputs
        query, context, metadata = self._preprocess(query, context, metadata)
        
        # Build prompt sections
        parts = [self.template.system_prompt]
        
        # Add metadata section if available and template supports it
        if metadata and self.template.metadata_template:
            metadata_str = self.template.format_metadata(metadata)
            if metadata_str:
                parts.append(metadata_str)
        
        # Add context section
        if context:
            context_str = self.template.format_context(context)
            parts.append(context_str)
        
        # Add query section
        query_str = self.template.format_query(query)
        parts.append(query_str)
        
        # Join sections
        prompt = self.template.separator.join(parts)
        
        # Postprocess
        prompt = self._postprocess(prompt)
        
        return prompt
    
    def build_with_history(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict],
        metadata: Optional[Dict] = None,
        max_history_turns: int = 3
    ) -> str:
        """
        Build prompt with explicit conversation history.
        
        Args:
            query: Current query
            context: Retrieved context
            conversation_history: List of previous turns
            metadata: Optional metadata
            max_history_turns: Maximum history turns to include
            
        Returns:
            Formatted prompt with history
        """
        # Format recent history
        history_str = ""
        recent_history = conversation_history[-max_history_turns:] if conversation_history else []
        
        for i, turn in enumerate(recent_history, 1):
            q = turn.get('query', turn.get('query_text', ''))
            r = turn.get('response', turn.get('response_text', ''))
            history_str += f"Turn {i}:\nUser: {q}\nAssistant: {r}\n\n"
        
        # Prepend history to context
        if history_str:
            full_context = f"# Recent Conversation\n{history_str}\n# Additional Context\n{context}"
        else:
            full_context = context
        
        return self.build(query, full_context, metadata)
    
    def build_multi_context(
        self,
        query: str,
        contexts: List[str],
        context_labels: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Build prompt with multiple labeled context sections.
        
        Args:
            query: User query
            contexts: List of context strings
            context_labels: Optional labels for each context
            metadata: Optional metadata
            
        Returns:
            Formatted prompt with multiple contexts
        """
        if not context_labels:
            context_labels = [f"Context {i+1}" for i in range(len(contexts))]
        
        # Format multiple contexts
        combined_context = ""
        for label, ctx in zip(context_labels, contexts):
            if ctx:
                combined_context += f"## {label}\n{ctx}\n\n"
        
        return self.build(query, combined_context.strip(), metadata)
    
    def estimate_tokens(self, prompt: str) -> int:
        """
        Estimate token count for prompt (rough approximation).
        
        Args:
            prompt: Prompt string
            
        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ≈ 4 characters
        return len(prompt) // 4
    
    def validate_prompt(self, prompt: str, max_tokens: int = 4096) -> Dict:
        """
        Validate prompt length and structure.
        
        Args:
            prompt: Prompt to validate
            max_tokens: Maximum allowed tokens
            
        Returns:
            Validation results dictionary
        """
        estimated_tokens = self.estimate_tokens(prompt)
        
        return {
            'valid': estimated_tokens <= max_tokens,
            'estimated_tokens': estimated_tokens,
            'max_tokens': max_tokens,
            'character_count': len(prompt),
            'has_system_prompt': self.template.system_prompt in prompt,
            'has_query': bool(prompt),
            'overflow_tokens': max(0, estimated_tokens - max_tokens)
        }
    
    def get_template_info(self) -> Dict:
        """
        Get information about current template.
        
        Returns:
            Template information dictionary
        """
        return {
            'template_name': self.template_name,
            'system_prompt_length': len(self.template.system_prompt),
            'has_metadata_support': self.template.metadata_template is not None,
            'max_context_length': self.template.max_context_length,
            'separator': repr(self.template.separator),
            'num_preprocessors': len(self.preprocessors),
            'num_postprocessors': len(self.postprocessors)
        }
    
    @staticmethod
    def create_custom_template(
        system_prompt: str,
        context_format: str = "Context:\n{context}",
        query_format: str = "Query: {query}\nResponse:",
        **kwargs
    ) -> PromptTemplate:
        """
        Factory method to create custom templates easily.
        
        Args:
            system_prompt: System instruction
            context_format: Context formatting string
            query_format: Query formatting string
            **kwargs: Additional PromptTemplate parameters
            
        Returns:
            New PromptTemplate instance
        """
        return PromptTemplate(
            system_prompt=system_prompt,
            context_template=context_format,
            query_template=query_format,
            **kwargs
        )


# Useful preprocessor functions
def trim_whitespace_preprocessor(query: str, context: str, metadata: Optional[Dict]) -> tuple:
    """Remove excessive whitespace from inputs."""
    return query.strip(), context.strip(), metadata


def lowercase_query_preprocessor(query: str, context: str, metadata: Optional[Dict]) -> tuple:
    """Convert query to lowercase (for case-insensitive systems)."""
    return query.lower(), context, metadata


def add_timestamp_preprocessor(query: str, context: str, metadata: Optional[Dict]) -> tuple:
    """Add timestamp to metadata."""
    if metadata is None:
        metadata = {}
    metadata['timestamp'] = datetime.now().isoformat()
    return query, context, metadata


# Useful postprocessor functions
def remove_extra_newlines_postprocessor(prompt: str) -> str:
    """Remove excessive newlines (more than 2 consecutive)."""
    import re
    return re.sub(r'\n{3,}', '\n\n', prompt)


def add_prompt_markers_postprocessor(prompt: str) -> str:
    """Add markers for prompt boundaries."""
    return f"<PROMPT_START>\n{prompt}\n<PROMPT_END>"


def truncate_prompt_postprocessor(max_length: int = 8000):
    """Create a postprocessor that truncates prompts to max length."""
    def truncate(prompt: str) -> str:
        if len(prompt) > max_length:
            return prompt[:max_length] + "\n[... prompt truncated ...]"
        return prompt
    return truncate


# Example usage and factory functions
def create_drift_aware_builder() -> PromptBuilder:
    """Create a prompt builder optimized for drift-aware conversations."""
    return PromptBuilder(
        template_name='drift_aware',
        preprocessors=[trim_whitespace_preprocessor, add_timestamp_preprocessor],
        postprocessors=[remove_extra_newlines_postprocessor]
    )


def create_research_builder() -> PromptBuilder:
    """Create a prompt builder optimized for research tasks."""
    return PromptBuilder(
        template_name='research',
        preprocessors=[trim_whitespace_preprocessor],
        postprocessors=[remove_extra_newlines_postprocessor]
    )


def create_code_builder() -> PromptBuilder:
    """Create a prompt builder optimized for code assistance."""
    return PromptBuilder(
        template_name='code_assistant',
        preprocessors=[trim_whitespace_preprocessor]
    )


__all__ = [
    'PromptTemplate',
    'PromptBuilder',
    'create_drift_aware_builder',
    'create_research_builder',
    'create_code_builder',
    'trim_whitespace_preprocessor',
    'add_timestamp_preprocessor',
    'remove_extra_newlines_postprocessor',
    'truncate_prompt_postprocessor'
]