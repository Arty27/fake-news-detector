from models.bert_classifier import BERTFakeNewsClassifier
from models.sentiment_analysis import SentimentAnalyzer
from models.named_entitiy_recognition import NERExtractor
from models.claim_density import ClaimDensityScore
from models.source_trust_checker import SourceTrustChecker
from models.live_checker import LiveCheckerService
from models.logical_inconsistency_checker import LogicalInconsistencyChecker

# text="Infinity stones found in Bangalore, Time stone found while digging for construction"
# text1 = "We faced hunger before, but never like this': skeletal children fill hospital wards as starvation grips Gaza"
# text = "Health experts warn that a new wave of infections may be linked to the widespread use of 5G towers in urban areas. Residents in several cities have reported symptoms including headaches, fatigue, and memory issues. One anonymous source stated that 'many believe the radiation is far more dangerous than reported.' While official health agencies deny any connection, growing public concern has prompted calls for investigation."
text = "The incredible hulk is one of the best left handed batsman in rugby"

classifier = BERTFakeNewsClassifier()
analyzer = SentimentAnalyzer()
ner = NERExtractor()
scorer = ClaimDensityScore()
source_checker = SourceTrustChecker()
logical_checker = LogicalInconsistencyChecker()

p_fake, p_real = classifier.predict(text)
print(f"Fake probability: {p_fake:.2f}, Real probability: {p_real:.2f}")
sentiment_result = analyzer.analyze(text)
print(sentiment_result)
entities = ner.extract(text)
print(entities)
print(scorer.score(text))

result = source_checker.check("https://www.abcnews.com.co/politics/trump-wins-2024/")
print(result)

article = "The incredible hulk is one of the best left handed batsman in rugby"
print(f"\n=== Testing Logical Inconsistency Checker ===")
print(f"Article: {article}")

# Check for logical inconsistencies
logical_result = logical_checker.check(article)
print(f"\nLogical Inconsistency Results:")
print(f"Inconsistency Count: {logical_result['inconsistency_count']}")
print(f"Overall Inconsistency Score: {logical_result['overall_inconsistency_score']:.3f}")
print(f"Is Logically Consistent: {logical_result['is_logically_consistent']}")
print(f"Confidence: {logical_result['confidence']:.3f}")

if logical_result['inconsistencies']:
    print(f"\nDetected Inconsistencies:")
    for i, inc in enumerate(logical_result['inconsistencies'], 1):
        print(f"{i}. Type: {inc.inconsistency_type}")
        print(f"   Description: {inc.description}")
        print(f"   Entities: {', '.join(inc.entities_involved)}")
        print(f"   Confidence: {inc.confidence:.3f}")
        print(f"   Rule: {inc.rule_applied}")
        print()

# Now test with the live checker
svc = LiveCheckerService()
result = svc.check(article)
print("Queries:", result["queries"])
print("Decision:", result["decision"])
for i, m in enumerate(result["top_matches"], 1):
    print(
        f"{i}. {m['similarity']:.3f} | {m['source']} | {m['title']} | {m['published_at']}"
    )

