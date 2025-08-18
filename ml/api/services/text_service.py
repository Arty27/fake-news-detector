import time
from datetime import datetime
from typing import Dict, Any, Optional
import sys
import os

# Add the parent directory to the path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.bert_classifier import BERTFakeNewsClassifier
from models.sentiment_analysis import SentimentAnalyzer
from models.named_entitiy_recognition import NERExtractor
from models.claim_density import ClaimDensityScore
from models.live_checker import LiveCheckerService
from models.comprehensive_fake_news_detector import ComprehensiveFakeNewsDetector


class TextAnalysisService:
    """Service for analyzing text articles using the fake news detection system"""

    def __init__(self):
        """Initialize all required models"""
        self.classifier = BERTFakeNewsClassifier()
        self.analyzer = SentimentAnalyzer()
        self.ner = NERExtractor()
        self.scorer = ClaimDensityScore()
        self.live_checker = LiveCheckerService()
        self.comprehensive_detector = ComprehensiveFakeNewsDetector()

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for fake news detection

        Args:
            text: Article text to analyze

        Returns:
            Complete analysis results formatted for API response
        """
        start_time = time.time()

        try:
            # Run all analyses
            p_fake, p_real = self.classifier.predict(text)
            sentiment_result = self.analyzer.analyze(text)
            entities = self.ner.extract(text)
            claim_result = self.scorer.score(text)
            live_result = self.live_checker.check(text)

            # Prepare data for comprehensive detection
            bert_data = {"fake_probability": p_fake, "real_probability": p_real}

            # Run comprehensive detection
            final_result = self.comprehensive_detector.detect(
                bert_data, sentiment_result, entities, claim_result, live_result
            )

            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)

            # Format response for API
            return {
                "final_verdict": final_result["final_verdict"],
                "confidence_level": final_result["confidence_level"],
                "fake_news_score": final_result["fake_news_score"],
                "reasoning": final_result["reasoning"],
                "factor_breakdown": final_result["factor_breakdown"],
                "dashboard_data": final_result["dashboard_data"],
                "analysis_timestamp": datetime.now(),
                "processing_time_ms": processing_time,
                # Add live checker results for dashboard display - use the original live_result
                "live_checker": live_result.get("top_matches", [{}]),
                # Add named entities for dashboard display - use the original entities
                "named_entities": entities if entities else [],
                "raw_data": {
                    "bert": bert_data,
                    "sentiment": sentiment_result,
                    "entities": entities,
                    "claim_density": claim_result,
                    "live_checker": live_result,
                },
            }

        except Exception as e:
            # Return error information
            return {
                "error": str(e),
                "analysis_timestamp": datetime.now(),
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

    def analyze_batch(self, texts: list[str]) -> list[Dict[str, Any]]:
        """
        Analyze multiple texts in batch

        Args:
            texts: List of article texts to analyze

        Returns:
            List of analysis results
        """
        results = []

        for text in texts:
            result = self.analyze_text(text)
            results.append(result)

        return results
