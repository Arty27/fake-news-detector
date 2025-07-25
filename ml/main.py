from models.bert_classifier import BERTFakeNewsClassifier
from models.sentiment_analysis import SentimentAnalyzer

text="Infinity stones found in Bangalore, Time stone found while digging for construction"
text1 = "We faced hunger before, but never like this': skeletal children fill hospital wards as starvation grips Gaza"

classifier = BERTFakeNewsClassifier()
analyzer = SentimentAnalyzer()
p_fake,p_real = classifier.predict(text)
print(f"Fake probability: {p_fake:.2f}, Real probability: {p_real:.2f}")
sentiment_result = analyzer.analyze(text1)
print(sentiment_result)