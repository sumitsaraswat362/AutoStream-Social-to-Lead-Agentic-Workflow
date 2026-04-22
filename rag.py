"""
RAG (Retrieval-Augmented Generation) module for AutoStream knowledge base.

Uses TF-IDF vectorization with cosine similarity for lightweight semantic search.
No external vector database required — runs entirely locally with zero infrastructure.

Design Decision:
    TF-IDF over raw keyword matching because it captures term importance
    (IDF weighting) and handles partial matches through n-gram overlap,
    while still being trivially deployable with no API costs or external services.
"""

import json
import os
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeBase:
    """
    Local knowledge base with TF-IDF-based semantic retrieval.

    Loads product data from a JSON file, flattens it into searchable text chunks,
    and provides similarity-based retrieval for RAG-powered responses.
    """

    def __init__(self, kb_path: Optional[str] = None):
        """
        Initialize the knowledge base from a JSON file.

        Args:
            kb_path: Path to knowledge_base.json. Defaults to same directory as this file.
        """
        if kb_path is None:
            kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.json")

        with open(kb_path, "r") as f:
            self.raw_data = json.load(f)

        # Build searchable chunks from structured data
        self.chunks = self._build_chunks()

        # Initialize TF-IDF vectorizer with bigrams for phrase matching
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),  # Unigrams + bigrams for better phrase matching
            max_features=5000,
        )

        # Pre-compute TF-IDF matrix for all chunks
        self.tfidf_matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

    def _build_chunks(self) -> list:
        """
        Flatten the structured JSON knowledge base into searchable text chunks.

        Each chunk contains:
            - text: The searchable content string
            - source: Category label (product_info, pricing, policy, faq)
            - metadata: Original structured data for reference
        """
        chunks = []

        # ── Product Overview ──────────────────────────────────────────
        product = self.raw_data.get("product", {})
        if product:
            chunks.append(
                {
                    "text": (
                        f"{product.get('name', '')} - {product.get('tagline', '')}. "
                        f"{product.get('description', '')}"
                    ),
                    "source": "product_info",
                    "metadata": product,
                }
            )

        # ── Pricing Plans ─────────────────────────────────────────────
        for plan in self.raw_data.get("plans", []):
            features_text = ", ".join(plan.get("features", []))
            chunk_text = (
                f"Pricing plan: {plan['name']} costs {plan['price']}. "
                f"Resolution: {plan.get('resolution', 'N/A')}. "
                f"Video limit: {plan.get('video_limit', 'N/A')}. "
                f"Features include: {features_text}"
            )
            chunks.append(
                {
                    "text": chunk_text,
                    "source": "pricing",
                    "metadata": plan,
                }
            )

        # ── Company Policies ──────────────────────────────────────────
        for policy_name, policy_text in self.raw_data.get("policies", {}).items():
            readable_name = policy_name.replace("_", " ").title()
            chunks.append(
                {
                    "text": f"{readable_name}: {policy_text}",
                    "source": "policy",
                    "metadata": {"policy": policy_name, "content": policy_text},
                }
            )

        # ── FAQ Entries ───────────────────────────────────────────────
        for faq in self.raw_data.get("faq", []):
            chunks.append(
                {
                    "text": f"Q: {faq['question']} A: {faq['answer']}",
                    "source": "faq",
                    "metadata": faq,
                }
            )

        return chunks

    def retrieve(
        self, query: str, top_k: int = 3, threshold: float = 0.05
    ) -> list:
        """
        Retrieve the most relevant knowledge chunks for a given query.

        Uses cosine similarity between TF-IDF vectors of the query and
        all stored chunks, returning the top-k matches above a minimum
        similarity threshold.

        Args:
            query: The user's natural language question
            top_k: Maximum number of results to return (default: 3)
            threshold: Minimum cosine similarity score to include (default: 0.05)

        Returns:
            List of dicts with keys: text, source, score, metadata
        """
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Rank by similarity, descending
        ranked_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            score = similarities[idx]
            if score >= threshold:
                results.append(
                    {
                        "text": self.chunks[idx]["text"],
                        "source": self.chunks[idx]["source"],
                        "score": float(score),
                        "metadata": self.chunks[idx]["metadata"],
                    }
                )

        return results

    def get_context_string(self, query: str, top_k: int = 3) -> str:
        """
        Get a formatted context string suitable for LLM prompting.

        Args:
            query: The user's question
            top_k: Number of chunks to include

        Returns:
            Formatted string with numbered source attributions
        """
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i} — {result['source']}]\n{result['text']}"
            )

        return "\n\n".join(context_parts)
