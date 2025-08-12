import os
from dotenv import load_dotenv

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
NEWSAPI_ENDPOINT = os.getenv("NEWSAPI_ENDPOINT", "https://newsapi.org/v2/everything")
NEWSAPI_LANGUAGE = os.getenv("NEWSAPI_LANGUAGE", "en")
NEWSAPI_PAGE_SIZE = int(os.getenv("NEWSAPI_PAGE_SIZE", "10"))

# Similarity
FRESHNESS_DECAY_DAYS = 14
