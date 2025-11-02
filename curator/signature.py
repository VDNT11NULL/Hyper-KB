import dspy

# DSPy signature definition
CuratorSignature = dspy.Signature("query_text, response_text -> keywords, entities, context_passage")
