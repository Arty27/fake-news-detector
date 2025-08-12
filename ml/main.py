from models.bert_classifier import BERTFakeNewsClassifier
from models.sentiment_analysis import SentimentAnalyzer
from models.named_entitiy_recognition import NERExtractor
from models.claim_density import ClaimDensityScore
from models.source_trust_checker import SourceTrustChecker
from models.live_checker import LiveCheckerService

# text="Infinity stones found in Bangalore, Time stone found while digging for construction"
# text1 = "We faced hunger before, but never like this': skeletal children fill hospital wards as starvation grips Gaza"
# text = "Health experts warn that a new wave of infections may be linked to the widespread use of 5G towers in urban areas. Residents in several cities have reported symptoms including headaches, fatigue, and memory issues. One anonymous source stated that 'many believe the radiation is far more dangerous than reported.' While official health agencies deny any connection, growing public concern has prompted calls for investigation."
text = "Small boat crossings to hit 50,000 since Labour came to power"

classifier = BERTFakeNewsClassifier()
analyzer = SentimentAnalyzer()
ner = NERExtractor()
scorer = ClaimDensityScore()
source_checker = SourceTrustChecker()

p_fake, p_real = classifier.predict(text)
print(f"Fake probability: {p_fake:.2f}, Real probability: {p_real:.2f}")
sentiment_result = analyzer.analyze(text)
print(sentiment_result)
entities = ner.extract(text)
print(entities)
print(scorer.score(text))

result = source_checker.check("https://www.abcnews.com.co/politics/trump-wins-2024/")
print(result)

article="Small boat crossings to hit 50,000 since Labour came to power"
print(f"\n=== Testing Live Checker ===")
print(f"Article: {article}")

# Now test with the live checker
svc = LiveCheckerService()
result = svc.check(article)
print("Queries:", result["queries"])
print("Decision:", result["decision"])
for i, m in enumerate(result["top_matches"], 1):
    print(
        f"{i}. {m['similarity']:.3f} | {m['source']} | {m['title']} | {m['published_at']}"
    )
