from dataclasses import dataclass
from typing import Optional


@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    description: Optional[str]
    published_at: Optional[str]
