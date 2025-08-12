from __future__ import annotations

import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from models.news_types import NewsItem

logger = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)


class SemanticReRanker:
    """
    Reranks candidate news items by cosine similarity to the user text.
    Baseline that uses title plus description for each item.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info("SentenceTransformer '%s' loaded for reranking.", model_name)

    def rerank(
        self,
        user_text: str,
        items: List[NewsItem],
        top_k: int = 10,
        show_progress: bool = False,
    ) -> List[Dict[str, Any]]:
        if not items:
            return []

        # Build texts to embed for candidates
        cand_texts = [(it.title or "") + " " + (it.description or "") for it in items]

        user_vec = self.model.encode([user_text], show_progress_bar=show_progress)
        item_vecs = self.model.encode(cand_texts, show_progress_bar=show_progress)
        sims = cosine_similarity(user_vec, item_vecs).flatten()

        scored: List[Dict[str, Any]] = []
        for it, sim in zip(items, sims):
            scored.append(
                {
                    "title": it.title,
                    "url": it.url,
                    "source": it.source,
                    "published_at": it.published_at,
                    "similarity": float(sim),
                }
            )

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]
