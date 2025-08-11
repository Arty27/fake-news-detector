from __future__ import annotations

import logging
from typing import List, Set

from models.news_types import NewsItem
from models.query_builder import QueryBuilder
from models.news_search_newsapi import NewsApiClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LiveSearchService:
    """
    Builds queries and fetches matching articles from NewsAPI.
    Deduplicates by URL or title.
    """

    def __init__(
        self, qb: QueryBuilder | None = None, client: NewsApiClient | None = None
    ):
        self.qb = qb or QueryBuilder()
        self.client = client or NewsApiClient()

    def search(
        self, article_text: str, page_size: int = 10
    ) -> tuple[list[str], list[NewsItem]]:
        queries = self.qb.build(article_text)
        all_items: List[NewsItem] = []
        seen: Set[str] = set()

        for q in queries:
            items = self.client.search(q, page_size=page_size)
            for it in items:
                key = it.url or it.title
                if key and key not in seen:
                    seen.add(key)
                    all_items.append(it)

        logger.info(
            "Collected %d unique items from %d queries.", len(all_items), len(queries)
        )
        return queries, all_items
