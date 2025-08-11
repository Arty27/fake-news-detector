from __future__ import annotations

import logging
from typing import Dict, Any

from models.live_search import LiveSearchService
from models.semantic_rerank import SemanticReRanker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LiveCheckerService:
    """
    Orchestrates query building, news search, and semantic re-ranking.
    Returns a compact JSON for the UI.
    """

    def __init__(
        self,
        search_service: LiveSearchService | None = None,
        reranker: SemanticReRanker | None = None,
    ):
        self.search_service = search_service or LiveSearchService()
        self.reranker = reranker or SemanticReRanker()

    def check(self, article_text: str) -> Dict[str, Any]:
        queries, items = self.search_service.search(article_text, page_size=10)
        ranked = self.reranker.rerank(
            article_text, items, top_k=10, show_progress=False
        )

        # Simple decision rule for now
        decision = "no corroboration found"
        if ranked:
            top_sim = ranked[0]["similarity"]
            if top_sim >= 0.80:
                decision = "strong corroboration found"
            elif top_sim >= 0.60:
                decision = "partial corroboration"

        return {
            "queries": queries,
            "top_matches": ranked[:5],
            "decision": decision,
        }
