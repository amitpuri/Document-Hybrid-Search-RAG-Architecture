"""
RAG Prompt Templates for Grounded Question Answering and Synthesis.
"""

SYSTEM_PROMPT = """You are an expert research assistant answering questions using technical scientific literature.
Your task is to provide accurate, grounded answers based EXCLUSIVELY on the provided context passages.

Guidelines:
1. Cite specific sources using their bracketed identifiers (e.g., [Source 1], [Source 2]).
2. Attribute technical concepts, theorems, or mechanisms to the specific paper and section where they appear.
3. If the context does not contain sufficient information to answer the question confidently, state clearly what is known and what is missing rather than hallucinating.
4. Synthesize multiple sources when comparing approaches or definitions.
"""


def format_qa_prompt(query: str, formatted_context: str) -> str:
    """Combines user query and formatted context into an instruction prompt."""
    return f"""Context Passages:
---------------------
{formatted_context}

User Question:
{query}

Grounded Answer:"""
