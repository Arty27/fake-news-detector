import os
import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from urllib.parse import urlparse
import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    AutoModel,
)
from sentence_transformers import SentenceTransformer
import spacy
from textblob import TextBlob
import tldextract
import validators
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class URLFakeNewsDetector:
    """
    Comprehensive fake news detection system using multiple AI techniques

    NOTE: This system uses a pattern-first approach that heavily weights obvious
    fake news patterns (fictional elements, impossible scenarios) while minimizing
    reliance on untrained models. It's designed to catch obvious fake news quickly
    and accurately.

    URL-ONLY VERSION: Only analyzes articles from URLs, not direct text input.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models
        self._load_models()

        # Trustworthy domains database
        self.trustworthy_domains = self._load_trustworthy_domains()

        # News API configuration
        self.news_api_key = os.getenv("NEWSAPI_KEY")

        # Meta classifier
        self.meta_classifier = None
        self._load_meta_classifier()

    def _load_models(self):
        """Load all required models"""
        logger.info("Loading models...")

        # BERT model for fake news detection - using lighter distilbert
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased"
            )
            self.bert_model.to(self.device)
            logger.info("BERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load BERT model: {e}")
            self.bert_model = None

        # Sentence transformer for embeddings - using very light model
        try:
            self.sentence_transformer = SentenceTransformer("paraphrase-MiniLM-L3-v2")
            logger.info("Sentence transformer loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_transformer = None

        # Spacy for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Spacy NER model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load Spacy model: {e}")
            self.nlp = None

        # Sentiment analysis pipeline - using lighter model
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
            )
            logger.info("Sentiment analysis model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load sentiment model: {e}")
            self.sentiment_analyzer = None

    def _load_trustworthy_domains(self) -> set:
        """Load database of trustworthy news domains"""
        trustworthy_domains = {
            "bbc.com",
            "bbc.co.uk",
            "reuters.com",
            "ap.org",
            "npr.org",
            "nytimes.com",
            "washingtonpost.com",
            "wsj.com",
            "cnn.com",
            "abcnews.go.com",
            "cbsnews.com",
            "nbcnews.com",
            "foxnews.com",
            "usatoday.com",
            "latimes.com",
            "chicagotribune.com",
            "bostonglobe.com",
            "theguardian.com",
            "independent.co.uk",
            "telegraph.co.uk",
            "lemonde.fr",
            "spiegel.de",
            "corriere.it",
            "elpais.com",
            "asahi.com",
            "scmp.com",
            "thehindu.com",
            "dawn.com",
        }
        return trustworthy_domains

    def _load_meta_classifier(self):
        """Load the meta classifier if available"""
        meta_classifier_path = os.path.join(self.models_dir, "meta_classifier.pkl")
        if os.path.exists(meta_classifier_path):
            try:
                self.meta_classifier = joblib.load(meta_classifier_path)
                logger.info("Meta classifier loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load meta classifier: {e}")

    def analyze_url(self, url: str) -> Dict:
        """
        Analyze article from URL for fake news indicators

        Args:
            url: URL to analyze

        Returns:
            Dictionary with analysis results
        """
        if not validators.url(url):
            return {
                "error": "Invalid URL format. Please provide a valid URL starting with http:// or https://"
            }

        try:
            # Extract article content
            article = Article(url)
            article.download()
            article.parse()

            if not article.text:
                # Fallback: Analyze the URL itself when text extraction fails
                logger.info(
                    f"Text extraction failed for {url}, falling back to URL analysis"
                )
                return self._analyze_url_fallback(url, article)

            # Analyze the extracted text using internal methods
            text_results = self._analyze_extracted_text(article.text)

            # Add URL-specific analysis
            url_results = {
                "url": url,
                "domain_trustworthiness": self._check_domain_trustworthiness(url),
                "article_title": article.title,
                "article_authors": article.authors,
                "article_publish_date": str(article.publish_date),
                "live_news_similarity": self._check_live_news_similarity(article.text),
            }

            # Combine results
            combined_results = {**text_results, **url_results}

            # Recalculate overall score with URL features
            combined_results["overall_score"] = self._calculate_overall_score(
                combined_results
            )
            combined_results["prediction"] = self._make_prediction(
                combined_results["overall_score"]
            )
            combined_results["confidence"] = self._calculate_confidence(
                combined_results
            )

            # Ensure all values are JSON serializable
            return self._ensure_json_serializable(combined_results)

        except requests.exceptions.ConnectionError as e:
            # Handle DNS resolution failures and connection issues
            if "NameResolutionError" in str(e) or "getaddrinfo failed" in str(e):
                logger.info(
                    f"DNS resolution failed for {url}, using fallback URL analysis"
                )
                return self._analyze_url_connection_fallback(
                    url, "DNS resolution failed - domain does not exist"
                )
            else:
                return {
                    "error": "Connection failed. The website might be down or blocking access."
                }
        except requests.exceptions.Timeout:
            return {
                "error": "Request timed out. The website is taking too long to respond."
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                return {
                    "error": "Access forbidden (403). This website blocks automated access."
                }
            elif e.response.status_code == 404:
                return {
                    "error": "Page not found (404). The URL might be broken or removed."
                }
            elif e.response.status_code == 429:
                return {
                    "error": "Too many requests (429). Please wait before trying again."
                }
            else:
                return {
                    "error": f"HTTP error {e.response.status_code}: {e.response.reason}"
                }
        except Exception as e:
            # Check for common connection-related errors
            error_str = str(e).lower()
            if any(
                keyword in error_str
                for keyword in ["dns", "getaddrinfo", "name resolution", "no such host"]
            ):
                logger.info(
                    f"DNS/connection error for {url}, using fallback URL analysis"
                )
                return self._analyze_url_connection_fallback(
                    url, "Connection failed - domain may not exist"
                )
            else:
                logger.error(f"Error analyzing URL {url}: {e}")
                return {"error": f"Failed to analyze URL: {str(e)}"}

    def _analyze_extracted_text(self, text: str) -> Dict:
        """
        Internal method to analyze extracted text - encapsulates core analysis logic
        """
        try:
            # BERT-based fake news detection
            bert_results = self._bert_fake_news_detection(text)

            # Sentiment analysis
            sentiment_results = self._sentiment_analysis(text)

            # Named Entity Recognition
            ner_results = self._named_entity_recognition(text)

            # Claim density scoring
            claim_density_results = self._claim_density_scoring(text)

            # Combine all results
            combined_results = {
                "bert_analysis": bert_results,
                "sentiment_analysis": sentiment_results,
                "named_entity_recognition": ner_results,
                "claim_density": claim_density_results,
            }

            # Calculate overall score
            combined_results["overall_score"] = self._calculate_overall_score(
                combined_results
            )
            combined_results["prediction"] = self._make_prediction(
                combined_results["overall_score"]
            )
            combined_results["confidence"] = self._calculate_confidence(
                combined_results
            )

            return combined_results

        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {"error": f"Text analysis failed: {str(e)}"}

    def _bert_fake_news_detection(self, text: str) -> Dict:
        """Detect fake news patterns using BERT"""
        if not self.bert_model:
            return {
                "score": 0.5,
                "confidence": 0.0,
                "error": "BERT model not available",
            }

        try:
            # Split text into chunks if too long
            max_length = 512
            chunks = [text[i : i + max_length] for i in range(0, len(text), max_length)]

            scores = []
            for chunk in chunks[:3]:  # Analyze first 3 chunks
                inputs = self.bert_tokenizer(
                    chunk, return_tensors="pt", truncation=True, max_length=max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    scores.append(probs[0][1].item())  # Probability of being fake

            avg_score = float(np.mean(scores)) if scores else 0.5
            return {
                "score": avg_score,
                "confidence": float(1.0 - np.std(scores)) if len(scores) > 1 else 0.8,
                "chunks_analyzed": len(scores),
            }

        except Exception as e:
            logger.error(f"BERT analysis error: {e}")
            return {"score": 0.5, "confidence": 0.0, "error": str(e)}

    def _sentiment_analysis(self, text: str) -> Dict:
        """Analyze sentiment and tone of the article"""
        if not self.sentiment_analyzer:
            return {
                "score": 0.0,
                "confidence": 0.0,
                "error": "Sentiment model not available",
            }

        try:
            # Split into sentences for better analysis
            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

            if not sentences:
                return {"score": 0.0, "confidence": 0.0, "error": "No sentences found"}

            # Analyze sentiment for each sentence
            sentiments = []
            for sentence in sentences[:10]:  # Limit to first 10 sentences
                if len(sentence) > 10:
                    result = self.sentiment_analyzer(sentence[:500])  # Limit length
                    if result:
                        label = result[0]["label"]
                        score = result[0]["score"]

                        # Convert to numerical score (-1 to 1)
                        if label == "NEGATIVE":
                            sentiments.append(-score)
                        elif label == "POSITIVE":
                            sentiments.append(score)
                        else:
                            sentiments.append(0)

            if not sentiments:
                return {
                    "score": 0.0,
                    "confidence": 0.0,
                    "error": "No sentiment results",
                }

            avg_sentiment = float(np.mean(sentiments))
            negative_ratio = float(
                len([s for s in sentiments if s < -0.3]) / len(sentiments)
            )

            return {
                "score": avg_sentiment,
                "negative_ratio": negative_ratio,
                "confidence": float(1.0 - np.std(sentiments)),
                "sentences_analyzed": len(sentiments),
            }

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"score": 0.0, "confidence": 0.0, "error": str(e)}

    def _named_entity_recognition(self, text: str) -> Dict:
        """Extract and analyze named entities"""
        if not self.nlp:
            return {
                "entities": [],
                "confidence": 0.0,
                "error": "NER model not available",
            }

        try:
            doc = self.nlp(text[:10000])  # Limit text length for performance

            entities = []
            entity_types = {}

            for ent in doc.ents:
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                )

                if ent.label_ not in entity_types:
                    entity_types[ent.label_] = 0
                entity_types[ent.label_] += 1

            return {
                "entities": entities,
                "entity_types": entity_types,
                "total_entities": len(entities),
                "confidence": 0.9 if entities else 0.5,
            }

        except Exception as e:
            logger.error(f"NER error: {e}")
            return {"entities": [], "confidence": 0.0, "error": str(e)}

    def _claim_density_scoring(self, text: str) -> Dict:
        """Calculate claim density and repetitiveness score"""
        if not self.sentence_transformer:
            return {
                "score": 0.5,
                "confidence": 0.0,
                "error": "Sentence transformer not available",
            }

        try:
            # Split into sentences
            sentences = [
                s.strip()
                for s in re.split(r"[.!?]+", text)
                if s.strip() and len(s.strip()) > 20
            ]

            if len(sentences) < 3:
                return {
                    "score": 0.5,
                    "confidence": 0.0,
                    "error": "Not enough sentences for analysis",
                }

            # Get embeddings for sentences
            embeddings = self.sentence_transformer.encode(sentences)

            # Calculate pairwise similarities
            similarity_matrix = cosine_similarity(embeddings)

            # Calculate average similarity (excluding self-similarity)
            similarities = []
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    similarities.append(similarity_matrix[i][j])

            if not similarities:
                return {
                    "score": 0.5,
                    "confidence": 0.0,
                    "error": "Could not calculate similarities",
                }

            avg_similarity = float(np.mean(similarities))
            similarity_std = float(np.std(similarities))

            # High similarity indicates repetitive content (potential fake news)
            repetitiveness_score = avg_similarity

            return {
                "score": repetitiveness_score,
                "confidence": float(1.0 - similarity_std),
                "sentences_analyzed": len(sentences),
                "avg_similarity": avg_similarity,
                "similarity_std": similarity_std,
            }

        except Exception as e:
            logger.error(f"Claim density error: {e}")
            return {"score": 0.5, "confidence": 0.0, "error": str(e)}

    def _check_domain_trustworthiness(self, url: str) -> Dict:
        """Check if the URL domain is trustworthy"""
        try:
            extracted = tldextract.extract(url)
            domain = f"{extracted.domain}.{extracted.suffix}"

            is_trustworthy = domain in self.trustworthy_domains

            # Additional checks
            additional_checks = {
                "has_https": url.startswith("https://"),
                "domain_age": "unknown",  # Could be enhanced with WHOIS lookup
                "social_media": any(
                    social in domain
                    for social in ["facebook", "twitter", "instagram", "tiktok"]
                ),
                "blog_platform": any(
                    platform in domain
                    for platform in ["blogspot", "wordpress", "medium", "substack"]
                ),
            }

            return {
                "domain": domain,
                "is_trustworthy": is_trustworthy,
                "trust_score": 1.0 if is_trustworthy else 0.2,
                "additional_checks": additional_checks,
            }

        except Exception as e:
            logger.error(f"Domain check error: {e}")
            return {"error": str(e), "trust_score": 0.0}

    def _check_live_news_similarity(self, text: str) -> Dict:
        """Check if the specific story exists in real news sources"""
        if not self.news_api_key:
            return {
                "score": 0.5,
                "confidence": 0.0,
                "error": "News API key not configured",
            }

        try:
            # Extract the core claim/story, not just random terms
            core_claim = self._extract_core_claim(text)
            if not core_claim:
                return {
                    "score": 0.5,
                    "confidence": 0.0,
                    "error": "Could not extract core claim",
                }

            # Search for the specific story, not just related terms
            search_query = f'"{core_claim}"'  # Exact phrase search
            try:
                url = f"https://newsapi.org/v2/everything"
                params = {
                    "q": search_query,
                    "apiKey": self.news_api_key,
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": 10,
                }

                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("articles", [])

                    # Check if any article actually contains the specific story
                    matching_stories = []
                    for article in articles:
                        if article.get("title") and article.get("description"):
                            article_text = (
                                f"{article['title']} {article['description']}"
                            )

                            # Check if the core claim is actually mentioned
                            if self._contains_core_claim(article_text, core_claim):
                                matching_stories.append(
                                    {
                                        "title": article["title"],
                                        "source": article.get("source", {}).get(
                                            "name", "Unknown"
                                        ),
                                        "url": article.get("url", ""),
                                        "published_at": article.get("publishedAt", ""),
                                    }
                                )

                    # Score based on whether the specific story exists
                    if matching_stories:
                        # Story found in real news = likely real
                        score = 0.1  # Very low fake news score
                        confidence = min(len(matching_stories) / 5.0, 1.0)
                    else:
                        # Story not found = potentially fake
                        score = 0.8  # High fake news score
                        confidence = 0.7

                    return {
                        "score": score,
                        "confidence": confidence,
                        "core_claim_searched": core_claim,
                        "matching_stories_found": len(matching_stories),
                        "stories": matching_stories[:5],
                    }

            except Exception as e:
                logger.warning(f"News API search error: {e}")
                return {"score": 0.5, "confidence": 0.0, "error": str(e)}

        except Exception as e:
            logger.error(f"Live news similarity error: {e}")
            return {"score": 0.5, "confidence": 0.0, "error": str(e)}

    def _extract_core_claim(self, text: str) -> str:
        """Extract the main claim/story from the text"""
        # Remove common filler words and focus on the core story
        text_lower = text.lower()

        # Look for the main action/subject
        if "found" in text_lower:
            # Extract "found X in Y" pattern
            match = re.search(r"found\s+([^.!?]+?)\s+in\s+([^.!?]+)", text_lower)
            if match:
                return f"found {match.group(1)} in {match.group(2)}"

        elif "makes" in text_lower and "debut" in text_lower:
            # Extract "X makes debut in Y" pattern
            match = re.search(r"(\w+)\s+makes\s+debut\s+in\s+([^.!?]+)", text_lower)
            if match:
                return f"{match.group(1)} makes debut in {match.group(2)}"

        elif "won" in text_lower:
            # Extract "X won Y" pattern
            match = re.search(r"(\w+)\s+won\s+([^.!?]+)", text_lower)
            if match:
                return f"{match.group(1)} won {match.group(2)}"

        # Fallback: return first sentence without common words
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if sentences:
            first_sentence = sentences[0]
            # Remove common filler words
            filler_words = [
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
            ]
            words = [
                word
                for word in first_sentence.split()
                if word.lower() not in filler_words
            ]
            return " ".join(words[:10])  # Limit to first 10 meaningful words

        return text[:100]  # Fallback to first 100 chars

    def _contains_core_claim(self, article_text: str, core_claim: str) -> bool:
        """Check if article actually contains the core claim"""
        article_lower = article_text.lower()
        claim_lower = core_claim.lower()

        # Check if key parts of the claim are present
        claim_words = [
            word for word in claim_lower.split() if len(word) > 3
        ]  # Only significant words

        # Need at least 70% of key words to match
        matches = sum(1 for word in claim_words if word in article_lower)
        match_ratio = matches / len(claim_words) if claim_words else 0

        return match_ratio >= 0.7

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        try:
            if not self.sentence_transformer:
                return 0.0

            embeddings = self.sentence_transformer.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)

        except Exception:
            return 0.0

    def _calculate_overall_score(self, results: Dict) -> float:
        """Calculate overall fake news score from individual feature scores"""
        scores = []
        weights = []

        # BERT analysis (reduced weight since it's less reliable for URLs)
        if "bert_analysis" in results and "score" in results["bert_analysis"]:
            scores.append(results["bert_analysis"]["score"])
            weights.append(0.1)

        # Sentiment analysis (only flag extreme negative cases)
        if (
            "sentiment_analysis" in results
            and "negative_ratio" in results["sentiment_analysis"]
        ):
            negative_ratio = results["sentiment_analysis"]["negative_ratio"]
            # Only contribute if sentiment is extremely negative (>80%)
            if negative_ratio > 0.8:
                scores.append(negative_ratio * 0.5)  # Reduced contribution
                weights.append(0.1)
            else:
                scores.append(0.1)  # Low score for normal sentiment
                weights.append(0.1)

        # Claim density (only flag extreme repetitiveness)
        if "claim_density" in results and "score" in results["claim_density"]:
            claim_score = results["claim_density"]["score"]
            # Only contribute if claim density is extremely high (>90%)
            if claim_score > 0.9:
                scores.append(claim_score * 0.3)  # Reduced contribution
                weights.append(0.1)
            else:
                scores.append(0.1)  # Low score for normal repetitiveness
                weights.append(0.1)

        # Domain trustworthiness (increased weight for URL analysis)
        if (
            "domain_trustworthiness" in results
            and "trust_score" in results["domain_trustworthiness"]
        ):
            trust_score = results["domain_trustworthiness"]["trust_score"]
            # Invert the score: low trust = high fake news probability
            fake_news_score = 1.0 - trust_score
            scores.append(fake_news_score)
            weights.append(0.4)

            # Add trust boost for highly trustworthy sources
            if trust_score > 0.8:
                scores.append(0.1)  # Very low fake news score
                weights.append(0.1)

        # Live news similarity (cross-reference verification)
        if (
            "live_news_similarity" in results
            and "score" in results["live_news_similarity"]
        ):
            similarity_score = results["live_news_similarity"]["score"]
            scores.append(similarity_score)
            weights.append(0.2)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.5  # Default neutral score

        weights = [w / total_weight for w in weights]

        # Calculate weighted average
        overall_score = sum(s * w for s, w in zip(scores, weights))

        return float(min(max(overall_score, 0.0), 1.0))

    def _make_prediction(self, score: float) -> str:
        """Make prediction based on score - balanced thresholds"""
        if score < 0.35:  # Balanced threshold
            return "likely_real"
        elif score < 0.65:  # Balanced threshold
            return "uncertain"
        else:
            return "likely_fake"

    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate confidence in the prediction"""
        confidences = []

        for key, value in results.items():
            if isinstance(value, dict) and "confidence" in value:
                confidences.append(value["confidence"])

        if not confidences:
            return 0.5

        return float(np.mean(confidences))

    def _ensure_json_serializable(self, obj):
        """Convert all NumPy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {
                key: self._ensure_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def train_meta_classifier(self, training_data: List[Dict], labels: List[int]):
        """Train the meta classifier with new data"""
        try:
            # Extract features
            features = []
            for data in training_data:
                feature_vector = [
                    data.get("bert_analysis", {}).get("score", 0.5),
                    data.get("sentiment_analysis", {}).get("negative_ratio", 0.5),
                    data.get("claim_density", {}).get("score", 0.5),
                    data.get("domain_trustworthiness", {}).get("trust_score", 0.5),
                    data.get("live_news_similarity", {}).get("score", 0.5),
                ]
                features.append(feature_vector)

            # Train classifier
            self.meta_classifier = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
            self.meta_classifier.fit(features, labels)

            # Save model
            os.makedirs(self.models_dir, exist_ok=True)
            model_path = os.path.join(self.models_dir, "meta_classifier.pkl")
            joblib.dump(self.meta_classifier, model_path)

            logger.info("Meta classifier trained and saved successfully")
            return True

        except Exception as e:
            logger.error(f"Error training meta classifier: {e}")
            return False

    def _analyze_url_fallback(self, url: str, article: Article) -> Dict:
        """
        Analyze URL when text extraction fails - provides fallback analysis
        based on URL patterns, domain trustworthiness, and other indicators
        """
        try:
            # Extract domain information
            domain_info = self._check_domain_trustworthiness(url)

            # Analyze URL patterns for suspicious indicators
            url_patterns = self._detect_suspicious_url_patterns(url)

            # Check if we got any metadata from the article
            has_metadata = bool(
                article.title or article.authors or article.publish_date
            )

            # Calculate fallback score based on URL analysis
            fallback_score = self._calculate_url_fallback_score(
                domain_info, url_patterns, has_metadata
            )

            # Create fallback results
            results = {
                "url": url,
                "analysis_type": "url_fallback",
                "reason": "Text extraction failed, analyzed URL patterns instead",
                "domain_trustworthiness": domain_info,
                "url_patterns": url_patterns,
                "article_metadata": {
                    "title": article.title,
                    "authors": article.authors,
                    "publish_date": (
                        str(article.publish_date) if article.publish_date else None
                    ),
                    "has_metadata": has_metadata,
                },
                "overall_score": fallback_score,
                "prediction": self._make_prediction(fallback_score),
                "confidence": 0.6,  # Lower confidence for URL-only analysis
                "warning": "⚠️ Analysis based on URL patterns only - content could not be extracted",
            }

            return self._ensure_json_serializable(results)

        except Exception as e:
            logger.error(f"URL fallback analysis failed: {e}")
            return {"error": f"URL analysis failed: {str(e)}"}

    def _analyze_url_connection_fallback(
        self, url: str, connection_reason: str
    ) -> Dict:
        """
        Analyze URL when connection fails - provides fallback analysis
        based on URL patterns and domain analysis without attempting to connect
        """
        try:
            # Extract domain information without connecting
            domain_info = self._check_domain_trustworthiness(url)

            # Analyze URL patterns for suspicious indicators
            url_patterns = self._detect_suspicious_url_patterns(url)

            # Check for obvious typos in domain names
            typo_analysis = self._detect_domain_typos(url)

            # Calculate fallback score based on URL analysis
            fallback_score = self._calculate_url_fallback_score(
                domain_info, url_patterns, False
            )

            # Adjust score based on typo detection
            if typo_analysis.get("likely_typo", False):
                fallback_score = min(
                    fallback_score + 0.3, 1.0
                )  # Increase fake news score for typos

            # Create fallback results
            results = {
                "url": url,
                "analysis_type": "url_connection_fallback",
                "reason": f"Connection failed: {connection_reason}",
                "domain_trustworthiness": domain_info,
                "url_patterns": url_patterns,
                "typo_analysis": typo_analysis,
                "overall_score": fallback_score,
                "prediction": self._make_prediction(fallback_score),
                "confidence": 0.7,  # Higher confidence for connection failures
                "warning": "⚠️ Analysis based on URL patterns only - could not connect to website",
                "connection_issue": connection_reason,
            }

            return self._ensure_json_serializable(results)

        except Exception as e:
            logger.error(f"URL connection fallback analysis failed: {e}")
            return {"error": f"URL analysis failed: {str(e)}"}

    def _detect_suspicious_url_patterns(self, url: str) -> Dict:
        """Detect suspicious patterns in URLs that might indicate fake news"""
        patterns = []
        red_flags = 0
        total_flags = 0

        url_lower = url.lower()

        # Pattern 1: Suspicious domains
        suspicious_domains = [
            "fake-news",
            "fake-news.com",
            "fake-news.org",
            "fake-news.net",
            "clickbait",
            "clickbait.com",
            "hoax",
            "hoax.com",
            "satire",
            "satire.com",
            "parody",
            "parody.com",
            "conspiracy",
            "conspiracy.com",
            "truth",
            "truth.com",
            "real-truth",
            "real-truth.com",
            "alternative-news",
            "alternative-news.com",
            "underground-news",
            "underground-news.com",
        ]

        for domain in suspicious_domains:
            if domain in url_lower:
                patterns.append("suspicious_domain_name")
                red_flags += 2
                break
        total_flags += 1

        # Pattern 2: URL structure patterns
        url_structure_patterns = [
            r"/\d{4}/\d{2}/\d{2}/",  # Date-based URLs (common in news)
            r"/\d{4}/\d{2}/",  # Year/month URLs
            r"/\d{4}/",  # Year-only URLs
            r"/[a-z]{3}-\d{4}/",  # Month-year URLs (e.g., jan-2024)
            r"/\d{8}/",  # 8-digit dates
        ]

        # Check if URL has proper news structure
        has_news_structure = any(
            re.search(pattern, url_lower) for pattern in url_structure_patterns
        )
        if not has_news_structure:
            patterns.append("missing_news_url_structure")
            red_flags += 1
        total_flags += 1

        # Pattern 3: Suspicious URL parameters
        suspicious_params = [
            "utm_source=fake",
            "utm_campaign=clickbait",
            "utm_medium=social",
            "ref=fake",
            "source=fake",
            "campaign=fake",
            "fbclid=fake",
            "gclid=fake",
            "msclkid=fake",
        ]

        for param in suspicious_params:
            if param in url_lower:
                patterns.append("suspicious_url_parameters")
                red_flags += 1
                break
        total_flags += 1

        # Pattern 4: URL length and complexity
        if len(url) > 200:
            patterns.append("excessively_long_url")
            red_flags += 1
        total_flags += 1

        # Pattern 5: Missing HTTPS
        if not url.startswith("https://"):
            patterns.append("missing_https")
            red_flags += 1
        total_flags += 1

        # Pattern 6: Generic/new domains
        domain_info = self._check_domain_trustworthiness(url)
        if not domain_info.get("is_trustworthy", False):
            # Check if it's a very new or generic domain
            if any(
                generic in url_lower
                for generic in ["news", "article", "story", "report"]
            ):
                patterns.append("generic_news_domain")
                red_flags += 1
        total_flags += 1

        # Pattern 7: Social media patterns (often unreliable for news)
        social_media_patterns = [
            "facebook.com",
            "twitter.com",
            "instagram.com",
            "tiktok.com",
            "reddit.com",
            "youtube.com",
            "linkedin.com",
        ]

        for social in social_media_patterns:
            if social in url_lower:
                patterns.append("social_media_source")
                red_flags += 1
                break
        total_flags += 1

        # Calculate pattern score
        pattern_score = red_flags / total_flags if total_flags > 0 else 0.0

        return {
            "patterns_detected": patterns,
            "red_flags": red_flags,
            "total_flags": total_flags,
            "score": min(pattern_score, 1.0),
            "has_news_structure": has_news_structure,
            "url_length": len(url),
            "uses_https": url.startswith("https://"),
            "confidence": 0.8 if patterns else 0.5,
        }

    def _calculate_url_fallback_score(
        self, domain_info: Dict, url_patterns: Dict, has_metadata: bool
    ) -> float:
        """Calculate fake news score based on URL analysis only"""
        scores = []
        weights = []

        # Domain trustworthiness (highest weight)
        domain_score = 1.0 - domain_info.get("trust_score", 0.0)
        scores.append(domain_score)
        weights.append(0.4)

        # URL pattern analysis
        if url_patterns.get("score", 0.0) > 0.3:
            scores.append(url_patterns["score"])
            weights.append(0.3)
        else:
            scores.append(0.2)  # Low score if no suspicious patterns
            weights.append(0.3)

        # Metadata availability
        if has_metadata:
            scores.append(0.1)  # Low fake news score if metadata exists
        else:
            scores.append(0.6)  # Higher score if no metadata
        weights.append(0.2)

        # URL structure
        if url_patterns.get("has_news_structure", False):
            scores.append(0.2)  # Low score for proper news URL structure
        else:
            scores.append(0.5)  # Medium score for missing structure
        weights.append(0.1)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Calculate weighted average
        overall_score = sum(s * w for s, w in zip(scores, weights))

        return float(min(max(overall_score, 0.0), 1.0))

    def _detect_domain_typos(self, url: str) -> Dict:
        """Detect potential typos in domain names that might indicate fake news"""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Common news domains and their common typos
            common_news_domains = {
                "theguardian.com": [
                    "theguaran.com",
                    "theguardian.co",
                    "theguardian.org",
                    "theguardian.net",
                ],
                "bbc.com": ["bbc.co", "bbc.org", "bbc.net", "bbcc.com"],
                "reuters.com": ["reuter.com", "reuters.co", "reuters.org"],
                "ap.org": ["ap.com", "ap.co", "ap.net"],
                "cnn.com": ["cnn.co", "cnn.org", "cnn.net"],
                "nytimes.com": ["nytime.com", "nytimes.co", "nytimes.org"],
                "washingtonpost.com": ["washingtonpost.co", "washingtonpost.org"],
                "wsj.com": ["wsj.co", "wsj.org", "wsj.net"],
                "bloomberg.com": ["bloomberg.co", "bloomberg.org"],
                "forbes.com": ["forbe.com", "forbes.co", "forbes.org"],
            }

            # Check if this domain is a typo of a known news source
            for real_domain, typos in common_news_domains.items():
                if domain in typos:
                    return {
                        "likely_typo": True,
                        "original_domain": real_domain,
                        "typo_type": "news_domain_typo",
                        "confidence": 0.9,
                        "suggestion": f"This appears to be a typo of {real_domain}",
                    }

            # Check for common typo patterns
            typo_patterns = [
                (r"\.co$", ".com"),  # .co instead of .com
                (r"\.org$", ".com"),  # .org instead of .com
                (r"\.net$", ".com"),  # .net instead of .com
                (r"\.info$", ".com"),  # .info instead of .com
            ]

            for pattern, correct in typo_patterns:
                if re.search(pattern, domain):
                    return {
                        "likely_typo": True,
                        "original_domain": domain.replace(
                            re.search(pattern, domain).group(), correct
                        ),
                        "typo_type": "tld_typo",
                        "confidence": 0.7,
                        "suggestion": f"Domain might be a typo of {domain.replace(re.search(pattern, domain).group(), correct)}",
                    }

            # Check for missing characters (common in typos)
            if len(domain) > 10:  # Only check longer domains
                for real_domain in common_news_domains.keys():
                    if len(real_domain) == len(domain) + 1:  # One character difference
                        # Simple similarity check
                        if self._calculate_string_similarity(domain, real_domain) > 0.8:
                            return {
                                "likely_typo": True,
                                "original_domain": real_domain,
                                "typo_type": "character_typo",
                                "confidence": 0.6,
                                "suggestion": f"Domain might be a typo of {real_domain}",
                            }

            return {"likely_typo": False, "confidence": 0.5}

        except Exception as e:
            logger.error(f"Domain typo detection failed: {e}")
            return {"likely_typo": False, "confidence": 0.3, "error": str(e)}

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity using character overlap"""
        if len(str1) == 0 or len(str2) == 0:
            return 0.0

        # Count matching characters in same positions
        matches = sum(1 for a, b in zip(str1, str2) if a == b)
        max_len = max(len(str1), len(str2))

        return matches / max_len


def main():
    """Command line interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Fake News Detector (URL-Only)")
    parser.add_argument("--url", help="URL to analyze", required=True)
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()

    detector = URLFakeNewsDetector()
    results = detector.analyze_url(args.url)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
