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
    
    def _extract_enhanced_entities(self, text: str) -> tuple[List[str], List[str]]:
        """Extract entities using both NER and custom patterns."""
        doc = self.nlp(text)
        
        # Standard NER entities
        names = [
            e.text for e in doc.ents if e.label_ in {"PERSON", "ORG", "GPE", "LOC"}
        ]
        dates_nums = [e.text for e in doc.ents if e.label_ in {"DATE", "CARDINAL"}]
        
        # Custom entity extraction for sports and fictional characters
        custom_entities = self._extract_custom_entities(text.lower())
        
        # Combine and deduplicate
        all_names = list(dict.fromkeys(names + custom_entities))
        all_dates_nums = list(dict.fromkeys(dates_nums))
        
        return all_names, all_dates_nums
    
    def _extract_custom_entities(self, text: str) -> List[str]:
        """Extract custom entities that NER might miss."""
        custom_entities = []
        
        # Fictional characters and superheroes
        fictional_patterns = [
            r'\b(incredible\s+hulk)\b',
            r'\b(harry\s+potter)\b',
            r'\b(sherlock\s+holmes)\b',
            r'\b(spiderman|spider\s+man)\b',
            r'\b(superman|super\s+man)\b',
            r'\b(batman|bat\s+man)\b',
            r'\b(iron\s+man)\b',
            r'\b(captain\s+america)\b',
            r'\b(luke\s+skywalker)\b',
            r'\b(gandalf)\b',
            r'\b(frodo)\b'
        ]
        
        for pattern in fictional_patterns:
            matches = re.findall(pattern, text)
            custom_entities.extend(matches)
        
        # Sports terms
        sports_patterns = [
            r'\b(cricket|worldcup|ipl|test\s+match|odi|t20)\b',
            r'\b(football|fifa|championship|world\s+cup|premier\s+league)\b',
            r'\b(basketball|nba)\b',
            r'\b(tennis|wimbledon|us\s+open|australian\s+open)\b',
            r'\b(rugby|rugby\s+union|rugby\s+league)\b',
            r'\b(olympics|olympic\s+games)\b',
            r'\b(batsman|bowler|wicket|goal|try|touchdown|ace|serve)\b'
        ]
        
        for pattern in sports_patterns:
            matches = re.findall(pattern, text)
            custom_entities.extend(matches)
        
        # Remove duplicates and return
        return list(dict.fromkeys(custom_entities))

    def build(self, text: str, max_queries: int = 3, max_len: int = 180) -> List[str]:
        text = text.strip()
        
        # Handle very short texts (single words or very short phrases)
        if len(text.split()) <= 3:
            return [text[:max_len]]
        
        # For longer texts, always go through the full query building process
        sentences = self._split_sentences(text)
        if not sentences:
            return [text[:max_len]]

        scored = [(self._score_sentence(s), s) for s in sentences]
        scored.sort(key=lambda x: x[0], reverse=True)

        top = [s for _, s in scored[:2]]  # pick top 2 sentences
        queries: List[str] = []
        seen: Set[str] = set()

        for sent in top:
            # Use enhanced entity extraction
            names, dates_nums = self._extract_enhanced_entities(sent)

            # Tier 1: quoted sentence (only if not too long)
            if len(sent) <= max_len:
                q1 = f'"{sent}"'
                if q1 not in seen:
                    seen.add(q1)
                    queries.append(q1)
                    if len(queries) >= max_queries:
                        return queries

            # Tier 2: entities with date or numbers (only if we have meaningful entities)
            if names or dates_nums:
                key_terms = names + dates_nums
                q2 = " ".join(dict.fromkeys(key_terms))
                if q2 and len(q2) > 2 and q2 not in seen:  # Ensure query is meaningful
                    seen.add(q2)
                    queries.append(q2)
                    if len(queries) >= max_queries:
                        return queries

            # Tier 3: entities only (only if we have meaningful entities)
            if names:
                q3 = " ".join(dict.fromkeys(names))
                if q3 and len(q3) > 2 and q3 not in seen:  # Ensure query is meaningful
                    seen.add(q3)
                    queries.append(q3)
                    if len(queries) >= max_queries:
                        return queries

        # If we still don't have enough meaningful queries, create better fallbacks
        if len(queries) < max_queries:
            # Extract meaningful keywords from the text
            meaningful_keywords = self._extract_meaningful_keywords(text)
            for keyword in meaningful_keywords:
                if keyword not in seen and len(keyword) > 2:
                    seen.add(keyword)
                    queries.append(keyword)
                    if len(queries) >= max_queries:
                        break

        return queries or [text[:max_len]]
    
    def _extract_meaningful_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text when NER fails."""
        # Remove common stop words and meaningless terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'one', 'two', 'three', 'first', 'second',
            'best', 'worst', 'good', 'bad', 'big', 'small', 'new', 'old', 'young', 'old', 'this', 'that',
            'these', 'those', 'here', 'there', 'where', 'when', 'why', 'how', 'what', 'who', 'which'
        }
        
        # Split text into words and filter meaningful ones
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        meaningful = []
        
        for word in words:
            # Skip stop words, single letters, and very short words
            if (word not in stop_words and 
                len(word) > 2 and 
                not word.isdigit() and
                word not in meaningful):
                meaningful.append(word)
        
        # Return top meaningful keywords (prioritize longer, more specific words)
        meaningful.sort(key=len, reverse=True)
        return meaningful[:5]  # Return top 5 meaningful keywords
