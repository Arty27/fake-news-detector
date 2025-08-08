from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ClaimDensityScore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Sentence Transformer Model '{model_name}' loaded.")

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def score(self, text: str) -> float:
        sentences = self._split_sentences(text)
        n = len(sentences)

        avg_sentence_length = sum(len(s.split()) for s in sentences) / n

        if n < 2:
            return 1.0  # Short text-> treat as high density

        embeddings = self.model.encode(sentences, show_progress_bar=False)
        similarity_matrix = cosine_similarity(embeddings)

        np.fill_diagonal(similarity_matrix, 0.0)

        avg_similarity = np.mean(similarity_matrix)
        density_score = 1 - avg_similarity

        if n < 6 and avg_sentence_length < 10:
            density_score *= 0.8

        return round(density_score, 4)
