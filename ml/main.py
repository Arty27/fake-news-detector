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
    print("🔍 TEXT-BASED FAKE NEWS DETECTION")
    print("=" * 50)
    
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
    
    # Display article information prominently
    print("📰 ARTICLE ANALYSIS")
    print("=" * 30)
    print(f"📝 Text Length: {len(text)} characters")
    print(f"🔍 Analysis Type: Direct Text Input")
    
    # Display named entities found in the text
    if entities:
        print(f"\n🏷️  NAMED ENTITIES IDENTIFIED:")
        print("=" * 35)
        entity_types = {}
        for entity in entities:
            entity_type = entity.get('label', 'UNKNOWN')
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity.get('text', ''))
        
        for entity_type, texts in entity_types.items():
            print(f"   {entity_type}: {', '.join(texts[:5])}")  # Show first 5 of each type
            if len(texts) > 5:
                print(f"      ... and {len(texts) - 5} more")
    
    # Display final verdict prominently
    print(f"\n🚨 FAKE NEWS DETECTION RESULTS")
    print("=" * 40)
    print(f"🎯 Final Verdict: {final_result['final_verdict']}")
    print(f"📊 Fake News Score: {final_result['fake_news_score']}")
    print(f"🎲 Confidence: {final_result['confidence_level']}")
    print(f"💡 Reasoning: {final_result['reasoning']}")
    
    # Display live news verification prominently
    if 'live_checker' in final_result['factor_breakdown']:
        live_data = final_result['factor_breakdown']['live_checker']
        print(f"\n🌐 LIVE NEWS VERIFICATION (NewsAPI)")
        print("=" * 40)
        print(f"📊 Verification Score: {live_data.get('score', 0):.3f}")
        print(f"🎯 Decision: {live_data.get('decision', 'N/A')}")
        print(f"🔍 Queries Generated: {live_data.get('queries_count', 0)}")
        print(f"📈 Top Similarity: {live_data.get('top_similarity', 0):.3f}")
        
        # Get the actual live checker results for more details
        live_results = live_result  # Use the live_result from the analysis
        if live_results and 'top_matches' in live_results:
            print(f"📰 Stories Found: {len(live_results.get('top_matches', []))}")
            
            # Display queries generated for verification
            if live_results.get('queries'):
                print(f"\n🔍 SEARCH QUERIES GENERATED:")
                for i, query in enumerate(live_results['queries'][:5], 1):  # Show top 5 queries
                    print(f"   {i}. \"{query}\"")
            
            # Display top matching stories if available
            if live_results.get('top_matches'):
                print(f"\n📋 TOP MATCHING STORIES FROM NEWS API:")
                for i, match in enumerate(live_results['top_matches'][:5], 1):  # Show top 5
                    print(f"   {i}. 📰 {match.get('title', 'No title')}")
                    print(f"      🏢 Source: {match.get('source', 'Unknown')}")
                    print(f"      🔗 URL: {match.get('url', 'No URL')}")
                    print(f"      📅 Date: {match.get('published_at', 'Unknown')}")
                    print(f"      📊 Similarity: {match.get('similarity', 0):.3f}")
                    print()
        else:
            print("   ⚠️  Live verification details not available")
    
    print(f"\n📋 DETAILED FACTOR BREAKDOWN:")
    for factor_name, factor_data in final_result['factor_breakdown'].items():
        print(f"   {factor_name.upper()}:")
        print(f"     Score: {factor_data.get('score', 0):.3f}")
        print(f"     Weight: {factor_data.get('weight', 0):.2f}")
        print(f"     Contribution: {factor_data.get('contribution', 0):.4f}")
        
        # Handle specific factors with additional details
        if factor_name == 'bert' and 'decision' in factor_data:
            print(f"     Decision: {factor_data['decision']}")
        
        elif factor_name == 'claim_density':
            print(f"     Semantic Density: {factor_data.get('semantic_density', 0):.3f}")
            print(f"     Claim Count: {factor_data.get('claim_count', 0)}")
        
        elif factor_name == 'named_entities':
            print(f"     Entities Found: {factor_data.get('entities_found', 0)}")
            print(f"     Entity Types: {', '.join(factor_data.get('entity_types', []))}")
        
        elif factor_name == 'sentiment':
            if 'sentiment_category' in factor_data:
                print(f"     Category: {factor_data['sentiment_category']}")
    
    print(f"\n🖥️  DASHBOARD DATA:")
    dashboard = final_result['dashboard_data']
    print(f"   Color Code: {dashboard['color']}")
    print(f"   Risk Level: {dashboard['summary']['risk_level'].upper()}")
    print(f"   Verdict Category: {dashboard['summary']['verdict_category']}")
    
    print("\n" + "=" * 50)
    print("✅ Text Analysis Complete!")

def analyze_url_article(url: str):
    """Analyze an article from URL using the URL fake news detector"""
    print("🔍 URL-BASED FAKE NEWS DETECTION")
    print("=" * 50)
    
    # Initialize URL detector
    detector = URLFakeNewsDetector()
    
    # Analyze the URL
    results = detector.analyze_url(url)
    
    # Check for errors
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Display article source and headline prominently
    print("📰 ARTICLE INFORMATION")
    print("=" * 30)
    print(f"🌐 Source URL: {results.get('url', 'N/A')}")
    print(f"📰 Headline: {results.get('article_title', 'N/A')}")
    print(f"👥 Authors: {', '.join(results.get('article_authors', ['N/A']))}")
    print(f"📅 Published: {results.get('article_publish_date', 'N/A')}")
    
    # Display domain trustworthiness
    if 'domain_trustworthiness' in results:
        domain_info = results['domain_trustworthiness']
        print(f"\n🔗 SOURCE CREDIBILITY:")
        print(f"   Domain: {domain_info.get('domain', 'N/A')}")
        print(f"   Trustworthy Source: {'✅ YES' if domain_info.get('is_trustworthy') else '❌ NO'}")
        print(f"   Trust Score: {domain_info.get('trust_score', 0):.3f}")
    
    # Display live news verification prominently
    if 'live_news_similarity' in results:
        live_info = results['live_news_similarity']
        print(f"\n🌐 LIVE NEWS VERIFICATION (NewsAPI)")
        print("=" * 40)
        print(f"🔍 Core Claim Searched: {live_info.get('core_claim_searched', 'N/A')}")
        print(f"📊 Verification Score: {live_info.get('score', 0):.3f}")
        print(f"🎯 Confidence: {live_info.get('confidence', 0):.3f}")
        print(f"📰 Stories Found: {live_info.get('matching_stories_found', 0)}")
        
        # Display matching stories if available
        if live_info.get('stories') and len(live_info['stories']) > 0:
            print(f"\n📋 MATCHING STORIES FROM NEWS API:")
            for i, story in enumerate(live_info['stories'][:5], 1):  # Show top 5
                print(f"   {i}. 📰 {story.get('title', 'No title')}")
                print(f"      🏢 Source: {story.get('source', 'Unknown')}")
                print(f"      🔗 URL: {story.get('url', 'No URL')}")
                print(f"      📅 Date: {story.get('published_at', 'Unknown')}")
                print()
        else:
            print("   ⚠️  No matching stories found in NewsAPI")
    
    # Display final verdict prominently
    print(f"\n🚨 FAKE NEWS DETECTION RESULTS")
    print("=" * 40)
    print(f"🎯 Final Verdict: {results.get('prediction', 'N/A').upper()}")
    print(f"📊 Fake News Score: {results.get('overall_score', 0):.3f}")
    print(f"🎲 Confidence: {results.get('confidence', 0):.3f}")
    
    # Display detailed analysis if available
    if 'bert_analysis' in results:
        bert_info = results['bert_analysis']
        print(f"\n🤖 BERT ANALYSIS:")
        print(f"   Score: {bert_info.get('score', 0):.3f}")
        print(f"   Confidence: {bert_info.get('confidence', 0):.3f}")
    
    if 'sentiment_analysis' in results:
        sentiment_info = results['sentiment_analysis']
        print(f"\n😊 SENTIMENT ANALYSIS:")
        print(f"   Score: {sentiment_info.get('score', 0):.3f}")
        print(f"   Negative Ratio: {sentiment_info.get('negative_ratio', 0):.3f}")
    
    if 'claim_density' in results:
        claim_info = results['claim_density']
        print(f"\n📊 CLAIM DENSITY:")
        print(f"   Score: {claim_info.get('score', 0):.3f}")
        print(f"   Avg Similarity: {claim_info.get('avg_similarity', 0):.3f}")
    
    print("\n" + "=" * 50)
    print("✅ URL Analysis Complete!")

def main():
    """Main function with choice between text and URL analysis"""
    print("🎯 UNIFIED FAKE NEWS DETECTION SYSTEM")
    print("=" * 50)
    print("Choose your analysis method:")
    print("1. Analyze text article")
    print("2. Analyze article from URL")
    print("3. Run demo examples")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        print("\n📝 Enter your article text (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        
        text = "\n".join(lines)
        if text.strip():
            analyze_text_article(text)
        else:
            print("❌ No text provided!")
    
    elif choice == "2":
        url = input("\n🌐 Enter the article URL: ").strip()
        if url:
            analyze_url_article(url)
        else:
            print("❌ No URL provided!")
    
    elif choice == "3":
        print("\n🚀 Running demo examples...")
        
        # Demo 1: Text analysis
        print("\n" + "="*60)
        demo_text = """
Thousands of people have been forced to leave their homes as deadly wildfires continue to burn across southern Europe amid a record-breaking heatwave.

Firefighters worked through the night to contain a blaze which broke out near Madrid in Spain.

A man caught in the fire in Tres Cantos, near the Spanish capital, died in hospital after suffering 98% burns.
"""
        analyze_text_article(demo_text)
        
        # Demo 2: URL analysis (using a known trustworthy source)
        print("\n" + "="*60)
        demo_url = "https://www.bbc.com/news/world-europe-68212345"
        print(f"🌐 Demo URL: {demo_url}")
        analyze_url_article(demo_url)
    
    else:
        print("❌ Invalid choice! Please run again and select 1, 2, or 3.")

if __name__ == "__main__":
    main()
