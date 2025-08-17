from typing import List, Dict
import spacy
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NERExtractor:

    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
            logger.info(f"Spacy model '{model}' loaded successfully!")
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model}' not found. Run: python -m spacy download {model}"
            )

    def extract(self, text: str) -> List[Dict[str, str]]:

        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string")

        doc = self.nlp(text)

        return [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
            if ent.label_ in {"PERSON", "ORG", "LOC", "GPE"}
        ]
