from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BERTFakeNewsClassifier:
    def __init__(self, model_name:str='mrm8488/bert-tiny-finetuned-fake-news-detection', device:str=None):
        """
        Initializes the model and tokenizer
        Args:
        model_name - Hugging face model name
        device - optional device overide (cuda or cpu)
        """
        self.device =  device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model= AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"BERT Model loaded on {self.device}")
    
    def predict(self, text:str)->Tuple[float, float]:
        if not text or not isinstance(text, str):
            raise ValueError("Input must be non empty string")

        inputs = self.tokenizer(text, return_tensors='pt', truncation = True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze().tolist()

        return probs[0], probs[1]  # (P(fake), P(real))