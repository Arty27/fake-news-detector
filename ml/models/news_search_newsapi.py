from __future__ import annotations

import logging
from typing import List, Optional
import requests
from tenacity import retry, wait_exponential, stop_after_attempt

from config import NEWSAPI_KEY, NEWSAPI_ENDPOINT, NEWSAPI_LANGUAGE, NEWSAPI_PAGE_SIZE
from models.news_types import NewsItem

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NewsApiClient:
    """
    Client for NewsAPI.org Everything endpoint.
    Returns a list of NewsItem objects.
    """

    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        self.api_key = api_key or NEWSAPI_KEY
        self.endpoint = endpoint or NEWSAPI_ENDPOINT
        if not self.api_key:
            raise ValueError("NEWSAPI_KEY is missing. Add it to your .env file.")

    @retry(
        wait=wait_exponential(multiplier=0.8, min=0.5, max=6),
        stop=stop_after_attempt(3),
    )
    def search(
        self,
        query: str,
        page_size: int = NEWSAPI_PAGE_SIZE,
        language: str = NEWSAPI_LANGUAGE,
    ) -> List[NewsItem]:
        params = {
            "q": query,
            "language": language,
            "pageSize": page_size,
            "sortBy": "publishedAt",
        }
        headers = {"X-Api-Key": self.api_key}

        resp = requests.get(self.endpoint, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            logger.warning(
                "NewsAPI returned status=%s for query=%s", data.get("status"), query
            )
            return []

        items: List[NewsItem] = []
        for a in data.get("articles", []):
            title = a.get("title") or ""
            url = a.get("url") or ""
            source_name = (a.get("source") or {}).get("name") or ""
            desc = a.get("description")
            published_at = a.get("publishedAt")
            items.append(
                NewsItem(
                    title=title,
                    url=url,
                    source=source_name,
                    description=desc,
                    published_at=published_at,
                )
            )

        logger.info("NewsAPI returned %d items for query '%s'", len(items), query)
        return items
