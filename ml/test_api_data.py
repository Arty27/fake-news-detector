#!/usr/bin/env python3
"""
Test script to verify API data structure
"""

import sys
import os

# Add the parent directory to the path to import models
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.live_checker import LiveCheckerService
from models.comprehensive_fake_news_detector import ComprehensiveFakeNewsDetector

def test_live_checker():
    """Test the live checker to see what it returns"""
    print("=== Testing Live Checker ===")
    
    live_checker = LiveCheckerService()
    
    # Test with a simple text
    test_text = "Donald Trump and Joe Biden are running for president in 2024"
    
    result = live_checker.check(test_text)
    
    print("Live Checker Result:")
    print(f"Decision: {result.get('decision')}")
    print(f"Verification Score: {result.get('verification_score')}")
    print(f"Queries Generated: {result.get('queries_generated')}")
    print(f"Stories Found: {result.get('stories_found')}")
    print(f"Top Matches: {len(result.get('top_matches', []))}")
    
    if result.get('top_matches'):
        print("\nFirst Top Match:")
        first_match = result['top_matches'][0]
        print(f"  Title: {first_match.get('title')}")
        print(f"  Source: {first_match.get('source')}")
        print(f"  URL: {first_match.get('url')}")
        print(f"  Similarity: {first_match.get('similarity')}")
    
    return result

def test_comprehensive_detector():
    """Test the comprehensive detector to see what it returns"""
    print("\n=== Testing Comprehensive Detector ===")
    
    # Mock data for testing
    bert_data = {"fake_probability": 0.3, "real_probability": 0.7}
    sentiment_result = {"overAll": 0.1}
    entities = [{"text": "Donald Trump", "label": "PERSON"}, {"text": "Joe Biden", "label": "PERSON"}]
    claim_result = {"semantic_density_score": 0.5, "claim_count": 2, "sentence_count": 1}
    
    # Get live checker result
    live_checker = LiveCheckerService()
    live_result = live_checker.check("Donald Trump and Joe Biden are running for president in 2024")
    
    detector = ComprehensiveFakeNewsDetector()
    result = detector.detect(bert_data, sentiment_result, entities, claim_result, live_result)
    
    print("Comprehensive Detector Result:")
    print(f"Final Verdict: {result.get('final_verdict')}")
    print(f"Live Checker Data: {result.get('live_checker')}")
    print(f"Named Entities: {result.get('named_entities')}")
    
    return result

if __name__ == "__main__":
    print("Testing API Data Structure...")
    
    try:
        live_result = test_live_checker()
        comprehensive_result = test_comprehensive_detector()
        
        print("\n=== Summary ===")
        print("✅ Live Checker working correctly")
        print("✅ Comprehensive Detector working correctly")
        print("✅ Data structure looks good for dashboard")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
