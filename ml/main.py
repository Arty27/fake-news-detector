from models.bert_classifier import BERTFakeNewsClassifier
from models.sentiment_analysis import SentimentAnalyzer
from models.named_entitiy_recognition import NERExtractor
from models.claim_density import ClaimDensityScore
from models.live_checker import LiveCheckerService
from models.comprehensive_fake_news_detector import ComprehensiveFakeNewsDetector
from models.url_fake_news_detector import URLFakeNewsDetector

import json

def analyze_text_article(text: str):
    """Analyze a text article using the comprehensive fake news detector"""
    # Initialize models
    classifier = BERTFakeNewsClassifier()
    analyzer = SentimentAnalyzer()
    ner = NERExtractor()
    scorer = ClaimDensityScore()
    live_checker = LiveCheckerService()
    comprehensive_detector = ComprehensiveFakeNewsDetector()

    # Run all analyses silently
    p_fake, p_real = classifier.predict(text)
    sentiment_result = analyzer.analyze(text)
    entities = ner.extract(text)
    claim_result = scorer.score(text)
    live_result = live_checker.check(text)
    
    # Prepare data for comprehensive detection
    bert_data = {"fake_probability": p_fake, "real_probability": p_real}
    
    # Run comprehensive detection
    final_result = comprehensive_detector.detect(
        bert_data, sentiment_result, entities, 
        claim_result, live_result
    )
    
    return final_result

def analyze_url_article(url: str):
    """Analyze an article from URL using the URL fake news detector"""
    # Initialize URL detector
    detector = URLFakeNewsDetector()
    
    # Analyze the URL
    results = detector.analyze_url(url)
    
    # Check for errors
    if "error" in results:
        return {"error": results['error']}
    
    return results

def main():
    """Main function - now clean and ready for API usage"""
    # This function is now clean and ready for API integration
    # All print statements have been removed
    pass

if __name__ == "__main__":
    main()
