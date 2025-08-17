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
        """Extract entities using NER and comprehensive keyword extraction."""
        doc = self.nlp(text)

        # Standard NER entities
        names = [
            e.text for e in doc.ents if e.label_ in {"PERSON", "ORG", "GPE", "LOC"}
        ]
        dates_nums = [e.text for e in doc.ents if e.label_ in {"DATE", "CARDINAL"}]

        # Extract important keywords that NER misses
        important_keywords = self._extract_important_keywords(text)

        # Combine and deduplicate
        all_names = list(dict.fromkeys(names + important_keywords))
        all_dates_nums = list(dict.fromkeys(dates_nums))

        return all_names, all_dates_nums

    def _extract_important_keywords(self, text: str) -> List[str]:
        """Extract important keywords using comprehensive NLP analysis."""
        # Clean the text first - remove quotes and apostrophes that cause issues
        clean_text = (
            text.replace("'", "").replace('"', "").replace('"', "").replace('"', "")
        )
        doc = self.nlp(clean_text.lower())

        important_terms = []

        # Extract nouns, verbs, adjectives, and proper nouns
        for token in doc:
            # Skip stop words, punctuation, and very short words
            if (
                not token.is_stop
                and not token.is_punct
                and not token.is_space
                and len(token.text) > 2
                and token.pos_ in {"NOUN", "VERB", "ADJ", "PROPN"}
            ):

                # Clean and add the word
                word = token.lemma_.lower()
                if word and word not in important_terms:
                    important_terms.append(word)

        # Extract noun phrases for better context
        noun_chunks = []
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            # Clean the chunk text
            clean_chunk = (
                chunk_text.replace("'", "")
                .replace('"', "")
                .replace('"', "")
                .replace('"', "")
            )
            if (
                len(clean_chunk) > 3
                and clean_chunk not in important_terms
                and clean_chunk not in noun_chunks
            ):
                noun_chunks.append(clean_chunk)

        # Extract verb phrases
        verb_phrases = []
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in {"ROOT", "ccomp", "xcomp"}:
                # Get the verb and its direct object if available
                for child in token.children:
                    if child.dep_ == "dobj" and not child.is_stop:
                        phrase = f"{token.lemma_.lower()} {child.lemma_.lower()}"
                        if phrase not in verb_phrases:
                            verb_phrases.append(phrase)

        # Combine all types of important terms
        all_terms = important_terms + noun_chunks + verb_phrases

        # Remove duplicates and return top terms
        unique_terms = list(dict.fromkeys(all_terms))
        return unique_terms[:15]  # Return top 15 terms

    def build(self, text: str, max_queries: int = 3, max_len: int = 180) -> List[str]:
        text = text.strip()

        # Handle very short texts
        if len(text.split()) <= 3:
            return [text[:max_len]]

        # Split into sentences
        sentences = self._split_sentences(text)
        if not sentences:
            return [text[:max_len]]

        # Extract entities from the entire text
        all_names, all_dates_nums = self._extract_enhanced_entities(text)

        queries = []
        seen = set()

        # Strategy 1: Create a comprehensive entity query
        if all_names:
            # Take top 4-5 most important entities
            key_entities = all_names[:5]
            entity_query = " ".join(key_entities)
            if len(entity_query) <= max_len and entity_query not in seen:
                seen.add(entity_query)
                queries.append(entity_query)

        # Strategy 2: Create focused entity queries (pairs)
        if all_names and len(queries) < max_queries:
            for i in range(0, len(all_names), 2):
                if len(queries) >= max_queries:
                    break
                if i + 1 < len(all_names):
                    pair_query = f"{all_names[i]} {all_names[i+1]}"
                else:
                    pair_query = all_names[i]

                if pair_query not in seen and len(pair_query) <= max_len:
                    seen.add(pair_query)
                    queries.append(pair_query)

        # Strategy 3: Create entity + number queries
        if all_names and all_dates_nums and len(queries) < max_queries:
            for entity in all_names[:3]:
                for number in all_dates_nums[:2]:
                    if len(queries) >= max_queries:
                        break
                    number_query = f"{entity} {number}"
                    if number_query not in seen and len(number_query) <= max_len:
                        seen.add(number_query)
                        queries.append(number_query)

        # Strategy 4: Use meaningful keywords as fallback
        if len(queries) < max_queries:
            meaningful_keywords = self._extract_meaningful_keywords(text)
            for keyword in meaningful_keywords:
                if len(queries) >= max_queries:
                    break
                if keyword not in seen and len(keyword) > 3:
                    seen.add(keyword)
                    queries.append(keyword)

        # Ensure we have at least some queries
        if not queries:
            # Last resort: use first few meaningful words
            words = [
                w
                for w in text.split()
                if len(w) > 3
                and w.lower() not in {"the", "and", "for", "with", "this", "that"}
            ]
            if words:
                fallback_query = " ".join(words[:4])
                queries.append(fallback_query[:max_len])

        return queries[:max_queries]

    def _extract_meaningful_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text when NER fails."""
        # Remove common stop words and meaningless terms
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "one",
            "two",
            "three",
            "first",
            "second",
            "best",
            "worst",
            "good",
            "bad",
            "big",
            "small",
            "new",
            "old",
            "young",
            "old",
            "this",
            "that",
            "these",
            "those",
            "here",
            "there",
            "where",
            "when",
            "why",
            "how",
            "what",
            "who",
            "which",
        }

        # Split text into words and filter meaningful ones
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        meaningful = []

        for word in words:
            # Skip stop words, single letters, and very short words
            if (
                word not in stop_words
                and len(word) > 2
                and not word.isdigit()
                and word not in meaningful
            ):
                meaningful.append(word)

        # Return top meaningful keywords (prioritize longer, more specific words)
        meaningful.sort(key=len, reverse=True)
        return meaningful[:5]  # Return top 5 meaningful keywords
