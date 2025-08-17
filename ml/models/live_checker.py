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
        verification_score = 0.0

        if ranked:
            top_sim = ranked[0]["similarity"]
            if top_sim >= 0.80:
                decision = "strong corroboration found"
                verification_score = 0.1  # Low fake news score for strong corroboration
            elif top_sim >= 0.60:
                decision = "partial corroboration"
                verification_score = (
                    0.3  # Medium fake news score for partial corroboration
                )
            else:
                verification_score = (
                    0.7  # Higher fake news score for weak corroboration
                )
        else:
            verification_score = 0.9  # High fake news score for no corroboration

        return {
            "queries": queries,
            "queries_generated": len(queries),
            "top_matches": ranked[:5],
            "decision": decision,
            "verification_score": verification_score,
            "stories_found": len(ranked),
        }
