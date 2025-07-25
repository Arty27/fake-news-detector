from typing import Dict
from vaderSentiment.vaderSentiment import  SentimentIntensityAnalyzer
import logging

logger=logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        logger.info("Vader Sentiment Analyzer initiated")

    def analyze(self, text:str)->Dict[str,float]:

        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string.")
        
        scores= self.analyzer.polarity_scores(text)

        return {
            "overAll":scores["compound"],
            "neg":scores["neg"],
            "nue":scores["neu"],
            "pos":scores["pos"]
        }