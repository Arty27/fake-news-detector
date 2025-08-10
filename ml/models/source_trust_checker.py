import json
import os
import logging
import tldextract
from typing import Dict, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SourceTrustChecker:
    def __init__(self, trust_list_path: str = "data/source_trust_list.json"):
        print(os.path.exists(trust_list_path))
        if not os.path.exists(trust_list_path):
            raise FileNotFoundError(
                f"Domain trust list not found at: {trust_list_path}"
            )

        with open(trust_list_path, "r") as file:
            data = json.load(file)
            self.trusted = data.get("trusted", {})
            self.untrusted = data.get("untrusted", {})
            logger.info("Source trust loaded.")

    def extract_domain(self, url: str) -> Optional[str]:
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
        return None

    def check(self, url: str) -> Dict[str, Optional[str]]:
        domain = self.extract_domain(url)
        if not domain:
            return {"domain": None, "label": "Unknown", "score": None}

        if domain in self.trusted:
            return {"domain": domain, "label": "trusted", "score": self.trusted[domain]}

        if domain in self.untrusted:
            return {
                "domain": domain,
                "label": "untrusted",
                "score": self.untrusted[domain],
            }

        return {"domain": domain, "label": "unknown", "score": None}
