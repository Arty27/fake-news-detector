import time
from datetime import datetime
from typing import Dict, Any, Optional
import sys
import os

# Add the parent directory to the path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.url_fake_news_detector import URLFakeNewsDetector

class URLAnalysisService:
    """Service for analyzing articles from URLs using the fake news detection system"""
    
    def __init__(self):
        """Initialize the URL detector"""
        self.detector = URLFakeNewsDetector()
    
    def analyze_url(self, url: str) -> Dict[str, Any]:
        """
        Analyze article from URL for fake news detection
        
        Args:
            url: URL of the article to analyze
            
        Returns:
            Complete analysis results formatted for API response
        """
        start_time = time.time()
        
        try:
            # Analyze the URL
            results = self.detector.analyze_url(url)
            
            # Check for errors
            if "error" in results:
                return {
                    "error": results["error"],
                    "analysis_timestamp": datetime.now(),
                    "processing_time_ms": int((time.time() - start_time) * 1000)
                }
            
            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            
            # Format response for API
            return {
                "final_verdict": results.get('prediction', 'unknown').upper(),
                "confidence_level": "HIGH" if results.get('confidence', 0) > 0.7 else "MEDIUM" if results.get('confidence', 0) > 0.4 else "LOW",
                "fake_news_score": results.get('overall_score', 0.5),
                "reasoning": f"Analysis based on URL patterns, domain trustworthiness, and content analysis",
                "factor_breakdown": self._format_factor_breakdown(results),
                "dashboard_data": self._prepare_dashboard_data(results),
                "analysis_timestamp": datetime.now(),
                "processing_time_ms": processing_time,
                # Add article information for dashboard display
                "article_source": results.get('domain_info', {}).get('domain', 'Unknown'),
                "article_headline": results.get('article_title', 'No Title Available'),
                "article_url": url,
                # Add live checker results for dashboard display
                "live_checker": self._format_live_checker_data(results),
                # Add named entities for dashboard display
                "named_entities": results.get('named_entities', []),
                "url_info": {
                    "url": results.get('url', 'N/A'),
                    "article_title": results.get('article_title', 'N/A'),
                    "article_authors": results.get('article_authors', []),
                    "article_publish_date": results.get('article_publish_date', 'N/A'),
                    "domain_trustworthiness": results.get('domain_trustworthiness', {}),
                    "live_news_similarity": results.get('live_news_similarity', {})
                },
                "raw_data": results
            }
            
        except Exception as e:
            # Return error information
            return {
                "error": str(e),
                "analysis_timestamp": datetime.now(),
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
    
    def _format_factor_breakdown(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format the factor breakdown for API response"""
        breakdown = {}
        
        # BERT analysis
        if 'bert_analysis' in results:
            bert_data = results['bert_analysis']
            breakdown['bert'] = {
                'score': bert_data.get('score', 0.0),
                'weight': 0.20,  # Updated weight from comprehensive detector
                'contribution': bert_data.get('score', 0.0) * 0.20,
                'decision': 'FAKE' if bert_data.get('score', 0.0) > 0.5 else 'REAL',
                'details': {
                    'confidence': bert_data.get('confidence', 0.0),
                    'chunks_analyzed': bert_data.get('chunks_analyzed', 0)
                }
            }
        
        # Live news similarity
        if 'live_news_similarity' in results:
            live_data = results['live_news_similarity']
            breakdown['live_checker'] = {
                'score': live_data.get('score', 0.0),
                'weight': 0.40,  # Updated weight from comprehensive detector
                'contribution': live_data.get('score', 0.0) * 0.40,
                'decision': live_data.get('decision', 'unknown'),
                'details': {
                    'core_claim_searched': live_data.get('core_claim_searched', 'N/A'),
                    'matching_stories_found': live_data.get('matching_stories_found', 0),
                    'stories': live_data.get('stories', [])
                }
            }
        
        # Claim density
        if 'claim_density' in results:
            claim_data = results['claim_density']
            breakdown['claim_density'] = {
                'score': claim_data.get('score', 0.0),
                'weight': 0.20,
                'contribution': claim_data.get('score', 0.0) * 0.20,
                'details': {
                    'avg_similarity': claim_data.get('avg_similarity', 0.0),
                    'similarity_std': claim_data.get('similarity_std', 0.0),
                    'sentences_analyzed': claim_data.get('sentences_analyzed', 0)
                }
            }
        
        # Named entities
        if 'named_entity_recognition' in results:
            ner_data = results['named_entity_recognition']
            breakdown['named_entities'] = {
                'score': 0.1 if ner_data.get('total_entities', 0) > 2 else 0.5,
                'weight': 0.10,
                'contribution': (0.1 if ner_data.get('total_entities', 0) > 2 else 0.5) * 0.10,
                'details': {
                    'entities_found': ner_data.get('total_entities', 0),
                    'entity_types': list(ner_data.get('entity_types', {}).keys())
                }
            }
        
        # Sentiment analysis
        if 'sentiment_analysis' in results:
            sentiment_data = results['sentiment_analysis']
            breakdown['sentiment'] = {
                'score': 0.1,  # Default low score for sentiment
                'weight': 0.10,
                'contribution': 0.1 * 0.10,
                'details': {
                    'overall_sentiment': sentiment_data.get('score', 0.0),
                    'negative_ratio': sentiment_data.get('negative_ratio', 0.0),
                    'sentences_analyzed': sentiment_data.get('sentences_analyzed', 0)
                }
            }
        
        return breakdown
    
    def _prepare_dashboard_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare dashboard data for API response"""
        overall_score = results.get('overall_score', 0.5)
        
        # Determine risk level and color
        if overall_score > 0.7:
            risk_level = 'high'
            color = '#dc3545'  # Red
        elif overall_score > 0.4:
            risk_level = 'medium'
            color = '#ffc107'  # Yellow
        else:
            risk_level = 'low'
            color = '#20c997'  # Green
        
        return {
            'risk_level': risk_level,
            'verdict_category': 'fake' if overall_score > 0.5 else 'real',
            'color': color,
            'confidence': results.get('confidence', 0.5)
        }
    
    def _format_live_checker_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format live checker data for dashboard display"""
        live_data = results.get('live_news_similarity', {})
        
        return {
            "decision": live_data.get('decision', 'Unknown'),
            "verification_score": live_data.get('score', 0.0),
            "queries_generated": live_data.get('queries_generated', 0),
            "top_matches": live_data.get('top_matches', []),
            "stories_found": len(live_data.get('top_matches', []))
        }
