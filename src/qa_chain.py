from __future__ import annotations

from typing import List, Tuple

from transformers import pipeline

from src.retriever import FAISSRetriever


class QAChain:
    def __init__(self, retriever: FAISSRetriever, model_name: str = "google/flan-t5-small"):
        self.retriever = retriever
        # Lazy load to avoid costly startup when module is imported.
        self._generator = pipeline("text2text-generation", model=model_name)

    def answer(self, query: str, top_k: int = 4) -> Tuple[str, List[dict]]:
        docs = self.retriever.similarity_search(query, k=top_k)
        if not docs:
            return "I could not find relevant information in the document.", []

        context_blocks = []
        citations = []
        for text, meta, _score in docs:
            context_blocks.append(text)
            citations.append(meta)

        prompt = self._format_prompt(query, context_blocks)
        response = self._generator(prompt, max_new_tokens=256, num_beams=4)[0]["generated_text"]
        return response.strip(), citations

    @staticmethod
    def _format_prompt(question: str, contexts: List[str]) -> str:
        joined_context = "\n\n".join(contexts)
        return (
            "You are a helpful assistant. Use the context to answer the question concisely. "
            "If the answer is not present, say you do not know.\n\n"
            f"Context:\n{joined_context}\n\nQuestion: {question}\nAnswer:"
        )
