from typing import List, Dict, Any
import logging
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ClaimDensityScore:
    """
    Claim density analysis using semantic similarity and basic claim classification.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Claim Density Model '{model_name}' loaded.")
    
    def score(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for claim density using semantic similarity and claim types.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return self._empty_result()
        
        # Classify claim types
        claim_types = [self._classify_claim_type(sentence) for sentence in sentences]
        
        # Calculate semantic density score (original algorithm)
        semantic_density_score = self._calculate_semantic_density_score(sentences)
        
        # Count claim types
        type_counts = {}
        for claim_type in claim_types:
            type_counts[claim_type] = type_counts.get(claim_type, 0) + 1
        
        return {
            "semantic_density_score": semantic_density_score,
            "claim_count": len(sentences),
            "claim_types": claim_types,
            "claim_type_distribution": type_counts,
            "sentence_count": len(sentences),
            "average_similarity": self._calculate_average_similarity(sentences)
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]
    
    def _classify_claim_type(self, sentence: str) -> str:
        """Classify the type of claim being made."""
        sentence_lower = sentence.lower()
        
        if re.search(r'\bwill\b|\bgoing\s+to\b|\bfuture\b|\bpredicted\b|\bforecast\b', sentence_lower):
            return "prediction"
        elif re.search(r'\bcauses\b|\bleads\s+to\b|\bbecause\b|\bdue\s+to\b|\bresults\s+in\b', sentence_lower):
            return "causal"
        elif re.search(r'\bbetter\b|\bworse\b|\bmore\b|\bless\b|\bsuperior\b|\binferior\b', sentence_lower):
            return "comparative"
        elif re.search(r'\b\d+%\b|\b\d+\s+(million|billion|thousand)\b|\bstatistics\b|\bdata\b', sentence_lower):
            return "statistical"
        elif re.search(r'\bI\s+think\b|\bbelieve\b|\bopinion\b|\bseems\b|\bappears\b', sentence_lower):
            return "opinion"
        elif re.search(r'\bproven\b|\bevidence\b|\bstudies\b|\bresearch\b|\bofficially\b', sentence_lower):
            return "factual"
        else:
            return "general"
    
    def _calculate_semantic_density_score(self, sentences: List[str]) -> float:
        """
        Calculate semantic density score using the original algorithm.
        High similarity = Low density (repetitive content)
        Low similarity = High density (diverse content)
        """
        if len(sentences) < 2:
            return 1.0  # Single sentence = maximum density
        
        try:
            # Encode sentences to embeddings
            embeddings = self.model.encode(sentences, show_progress_bar=False)
            
            # Calculate cosine similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Remove diagonal (self-similarity = 1.0)
            np.fill_diagonal(similarity_matrix, 0.0)
            
            # Calculate average similarity
            avg_similarity = np.mean(similarity_matrix)
            
            # Density score: 1 - average_similarity
            # High similarity = Low density (repetitive)
            # Low similarity = High density (diverse)
            density_score = 1.0 - avg_similarity
            
            return round(max(0.0, min(1.0, density_score)), 4)
            
        except Exception as e:
            logger.warning(f"Semantic analysis failed: {e}. Falling back to simple density.")
            return len(sentences) / max(len(sentences), 1)
    
    def _calculate_average_similarity(self, sentences: List[str]) -> float:
        """Calculate average similarity between all sentence pairs."""
        if len(sentences) < 2:
            return 0.0
        
        try:
            embeddings = self.model.encode(sentences, show_progress_bar=False)
            similarity_matrix = cosine_similarity(embeddings)
            np.fill_diagonal(similarity_matrix, 0.0)
            return round(float(np.mean(similarity_matrix)), 4)
        except Exception as e:
            logger.warning(f"Average similarity calculation failed: {e}")
            return 0.0
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            "semantic_density_score": 0.0,
            "claim_count": 0,
            "claim_types": [],
            "claim_type_distribution": {},
            "sentence_count": 0,
            "average_similarity": 0.0
        }
