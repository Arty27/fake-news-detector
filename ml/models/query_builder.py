from __future__ import annotations

import logging
import re
from typing import List, Set
import spacy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QueryBuilder:
    """
    Builds 2 to 3 short queries from a long article.
    Uses entities to keep the query specific.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info("spaCy model '%s' loaded for query building.", spacy_model)
        except OSError as e:
            raise RuntimeError(
                f"spaCy model '{spacy_model}' not found. Run: python -m spacy download {spacy_model}"
            ) from e

    def _split_sentences(self, text: str) -> List[str]:
        # Lightweight splitter works well enough for queries
        sents = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 0]
        return sents

    def _score_sentence(self, sent: str) -> int:
        doc = self.nlp(sent)
        ents = [
            e
            for e in doc.ents
            if e.label_ in {"PERSON", "ORG", "GPE", "LOC", "DATE", "CARDINAL"}
        ]
        score = len(ents)
        if any(e.label_ in {"DATE", "CARDINAL"} for e in ents):
            score += 1
        word_len = len(sent.split())
        if 8 <= word_len <= 30:
            score += 1
        return score

    def build(self, text: str, max_queries: int = 3, max_len: int = 180) -> List[str]:
        text = text.strip()
        if len(text) <= 100:
            return [text[:max_len]]

        sentences = self._split_sentences(text)
        if not sentences:
            return [text[:max_len]]

        scored = [(self._score_sentence(s), s) for s in sentences]
        scored.sort(key=lambda x: x[0], reverse=True)

        top = [s for _, s in scored[:2]]  # pick top 2 sentences
        queries: List[str] = []
        seen: Set[str] = set()

        for sent in top:
            doc = self.nlp(sent)
            names = [
                e.text for e in doc.ents if e.label_ in {"PERSON", "ORG", "GPE", "LOC"}
            ]
            dates_nums = [e.text for e in doc.ents if e.label_ in {"DATE", "CARDINAL"}]

            # Tier 1: quoted sentence
            q1 = f'"{sent}"'
            # Tier 2: entities with date or numbers
            key_terms = names + dates_nums
            q2 = " ".join(dict.fromkeys(key_terms))
            # Tier 3: entities only
            q3 = " ".join(dict.fromkeys(names))

            for q in (q1, q2, q3):
                q = q.strip()
                if not q:
                    continue
                if len(q) > max_len:
                    q = q[:max_len]
                if q not in seen:
                    seen.add(q)
                    queries.append(q)
                if len(queries) >= max_queries:
                    return queries

        return queries or [text[:max_len]]
