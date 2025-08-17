from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ComprehensiveFakeNewsDetector:
    """
    Comprehensive fake news detector that combines all analysis factors
    with appropriate weights to provide final verdicts.
    """
    
    def __init__(self):
        # Weights for different factors - Updated to prioritize live verification
        self.weights = {
            'bert_fake': 0.20,        # BERT when fake > real (reduced weight)
            'bert_real': 0.0,         # BERT when real > fake (ignored)
            'live_checker': 0.40,     # Live news verification (increased weight)
            'claim_density': 0.20,    # Claim density analysis
            'named_entities': 0.10,   # Named entity presence
            'sentiment': 0.10         # Sentiment analysis
        }
        
        # Thresholds for different verdicts
        self.thresholds = {
            'definitely_fake': 0.75,
            'likely_fake': 0.60,
            'suspicious': 0.45,
            'likely_real': 0.30,
            'definitely_real': 0.15
        }
    
    def detect(self, 
               bert_result: Dict[str, float],
               sentiment_result: Dict[str, float],
               ner_result: List[Dict[str, str]],
               claim_density_result: Dict[str, Any],
               live_checker_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive fake news detection using all available factors.
        """
        
        # Initialize scores
        total_score = 0.0
        factor_scores = {}
        factor_details = {}
        
        # Factor 1: BERT Classification (only when fake > real)
        bert_score = self._analyze_bert(bert_result)
        total_score += bert_score * self.weights['bert_fake']
        factor_scores['bert'] = bert_score
        factor_details['bert'] = {
            'score': bert_score,
            'weight': self.weights['bert_fake'],
            'contribution': bert_score * self.weights['bert_fake'],
            'decision': 'FAKE' if bert_result['fake_probability'] > bert_result['real_probability'] else 'REAL',
            'confidence': max(bert_result['fake_probability'], bert_result['real_probability'])
        }
        
        # Factor 2: Live News Verification
        live_score = self._analyze_live_checker(live_checker_result)
        total_score += live_score * self.weights['live_checker']
        factor_scores['live_checker'] = live_score
        factor_details['live_checker'] = {
            'score': live_score,
            'weight': self.weights['live_checker'],
            'contribution': live_score * self.weights['live_checker'],
            'decision': live_checker_result.get('decision', 'unknown'),
            'queries_count': len(live_checker_result.get('queries', [])),
            'top_similarity': live_checker_result.get('top_matches', [{}])[0].get('similarity', 0.0) if live_checker_result.get('top_matches') else 0.0
        }
        
        # Factor 3: Claim Density Analysis
        density_score = self._analyze_claim_density(claim_density_result)
        total_score += density_score * self.weights['claim_density']
        factor_scores['claim_density'] = density_score
        factor_details['claim_density'] = {
            'score': density_score,
            'weight': self.weights['claim_density'],
            'contribution': density_score * self.weights['claim_density'],
            'semantic_density': claim_density_result.get('semantic_density_score', 0.0),
            'claim_count': claim_density_result.get('claim_count', 0),
            'sentence_count': claim_density_result.get('sentence_count', 0)
        }
        
        # Factor 4: Named Entity Recognition
        ner_score = self._analyze_named_entities(ner_result)
        total_score += ner_score * self.weights['named_entities']
        factor_scores['named_entities'] = ner_score
        factor_details['named_entities'] = {
            'score': ner_score,
            'weight': self.weights['named_entities'],
            'contribution': ner_score * self.weights['named_entities'],
            'entities_found': len(ner_result) if ner_result else 0,
            'entity_types': list(set(entity['label'] for entity in ner_result)) if ner_result else []
        }
        
        # Factor 5: Sentiment Analysis
        sentiment_score = self._analyze_sentiment(sentiment_result)
        total_score += sentiment_score * self.weights['sentiment']
        factor_scores['sentiment'] = sentiment_score
        factor_details['sentiment'] = {
            'score': sentiment_score,
            'weight': self.weights['sentiment'],
            'contribution': sentiment_score * self.weights['sentiment'],
            'overall_sentiment': sentiment_result.get('overAll', 0.0),
            'sentiment_category': self._categorize_sentiment(sentiment_result.get('overAll', 0.0))
        }
        
        # Determine final verdict
        verdict, confidence, reasoning = self._determine_verdict(total_score, factor_details)
        
        # Apply logical consistency checks
        final_verdict, final_confidence, final_reasoning = self._apply_logical_checks(
            verdict, confidence, reasoning, total_score, factor_details
        )
        
        return {
            'final_verdict': final_verdict,
            'confidence_level': final_confidence,
            'fake_news_score': round(total_score, 4),
            'reasoning': final_reasoning,
            'factor_breakdown': factor_details,
            'factor_scores': factor_scores,
            'total_weighted_score': round(total_score, 4),
            'dashboard_data': self._prepare_dashboard_data(final_verdict, final_confidence, total_score, factor_details)
        }
    
    def _analyze_bert(self, bert_result: Dict[str, float]) -> float:
        """Analyze BERT results (only consider when fake > real)."""
        fake_prob = bert_result.get('fake_probability', 0.0)
        real_prob = bert_result.get('real_probability', 0.0)
        
        if fake_prob > real_prob:
            return fake_prob  # Return fake probability as score
        else:
            return 0.0  # Ignore when real > fake
    
    def _analyze_live_checker(self, live_result: Dict[str, Any]) -> float:
        """Analyze live news verification results."""
        decision = live_result.get('decision', '').lower()
        
        if 'strong corroboration' in decision:
            return 0.0  # Strong evidence of real news
        elif 'partial corroboration' in decision:
            # Check if the partial corroboration is strong enough
            top_similarity = live_result.get('top_matches', [{}])[0].get('similarity', 0.0) if live_result.get('top_matches') else 0.0
            stories_found = len(live_result.get('top_matches', []))
            
            # If similarity is high (>0.6) and multiple stories found, treat as strong evidence
            if top_similarity > 0.6 and stories_found >= 3:
                return 0.1  # Strong partial evidence = low fake news score
            else:
                return 0.3  # Weak partial evidence = moderately suspicious
        elif 'no corroboration' in decision:
            return 0.9  # No supporting evidence = highly suspicious/fake
        else:
            return 0.5  # Unknown status
    
    def _analyze_claim_density(self, density_result: Dict[str, Any]) -> float:
        """Analyze claim density results."""
        semantic_density = density_result.get('semantic_density_score', 0.0)
        claim_count = density_result.get('claim_count', 0)
        sentence_count = density_result.get('sentence_count', 0)
        
        # High density + low entities = likely fake
        if semantic_density > 0.8 and claim_count > 0:
            return 0.7  # Suspicious
        elif semantic_density > 0.6:
            return 0.4  # Moderately suspicious
        else:
            return 0.1  # Low suspicion
    
    def _analyze_named_entities(self, ner_result: List[Dict[str, str]]) -> float:
        """Analyze named entity presence."""
        if not ner_result:
            return 0.8  # No entities = highly suspicious
        
        total_entities = len(ner_result)
        
        if total_entities == 0:
            return 0.8  # No entities = highly suspicious
        elif total_entities <= 2:
            return 0.5  # Few entities = moderately suspicious
        else:
            return 0.1  # Many entities = low suspicion
    
    def _analyze_sentiment(self, sentiment_result: Dict[str, float]) -> float:
        """Analyze sentiment analysis results."""
        overall_sentiment = sentiment_result.get('overAll', 0.0)
        
        # Extreme negative sentiment might indicate fake news
        if overall_sentiment < -0.8:
            return 0.6  # Highly negative = suspicious
        elif overall_sentiment < -0.5:
            return 0.4  # Negative = moderately suspicious
        elif overall_sentiment > 0.8:
            return 0.2  # Highly positive = less suspicious
        else:
            return 0.1  # Neutral = low suspicion
    
    def _categorize_sentiment(self, sentiment_value: float) -> str:
        """Categorize sentiment value."""
        if sentiment_value < -0.5:
            return 'negative'
        elif sentiment_value > 0.5:
            return 'positive'
        else:
            return 'neutral'
    
    def _determine_verdict(self, total_score: float, factor_details: Dict[str, Any]) -> tuple[str, str, str]:
        """Determine final verdict based on total score."""
        
        if total_score >= self.thresholds['definitely_fake']:
            verdict = "DEFINITELY FAKE"
            confidence = "VERY HIGH"
            reasoning = "Multiple strong indicators suggest this is fake news"
        elif total_score >= self.thresholds['likely_fake']:
            verdict = "LIKELY FAKE"
            confidence = "HIGH"
            reasoning = "Several indicators suggest this is likely fake news"
        elif total_score >= self.thresholds['suspicious']:
            verdict = "SUSPICIOUS"
            confidence = "MEDIUM"
            reasoning = "Some concerning indicators, requires verification"
        elif total_score >= self.thresholds['likely_real']:
            verdict = "LIKELY REAL"
            confidence = "MEDIUM"
            reasoning = "Most indicators suggest this is likely real news"
        else:
            verdict = "DEFINITELY REAL"
            confidence = "VERY HIGH"
            reasoning = "Strong evidence suggests this is real news"
        
        return verdict, confidence, reasoning
    
    def _prepare_dashboard_data(self, verdict: str, confidence: str, score: float, factor_details: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data specifically for dashboard display."""
        
        # Color coding for dashboard
        color_map = {
            'DEFINITELY FAKE': '#dc3545',      # Red
            'LIKELY FAKE': '#fd7e14',          # Orange
            'SUSPICIOUS': '#ffc107',           # Yellow
            'LIKELY REAL': '#20c997',          # Teal
            'DEFINITELY REAL': '#198754'       # Green
        }
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'score': score,
            'color': color_map.get(verdict, '#6c757d'),
            'factors': [
                {
                    'name': 'BERT Classification',
                    'score': factor_details.get('bert', {}).get('score', 0),
                    'contribution': factor_details.get('bert', {}).get('contribution', 0),
                    'decision': factor_details.get('bert', {}).get('decision', 'N/A')
                },
                {
                    'name': 'Live Verification',
                    'score': factor_details.get('live_checker', {}).get('score', 0),
                    'contribution': factor_details.get('live_checker', {}).get('contribution', 0),
                    'decision': factor_details.get('live_checker', {}).get('decision', 'N/A')
                },
                {
                    'name': 'Claim Density',
                    'score': factor_details.get('claim_density', {}).get('score', 0),
                    'contribution': factor_details.get('claim_density', {}).get('contribution', 0),
                    'details': f"Density: {factor_details.get('claim_density', {}).get('semantic_density', 0):.2f}"
                },
                {
                    'name': 'Named Entities',
                    'score': factor_details.get('named_entities', {}).get('score', 0),
                    'contribution': factor_details.get('named_entities', {}).get('contribution', 0),
                    'details': f"Found: {factor_details.get('named_entities', {}).get('entities_found', 0)}"
                },
                {
                    'name': 'Sentiment',
                    'score': factor_details.get('sentiment', {}).get('score', 0),
                    'contribution': factor_details.get('sentiment', {}).get('contribution', 0),
                    'details': factor_details.get('sentiment', {}).get('sentiment_category', 'N/A')
                }
            ],
            'summary': {
                'total_score': score,
                'verdict_category': verdict.split()[0].lower(),  # 'definitely', 'likely', 'suspicious'
                'risk_level': 'high' if score > 0.6 else 'medium' if score > 0.3 else 'low'
            }
        }
    
    def _apply_logical_checks(self, verdict: str, confidence: str, reasoning: str, 
                             total_score: float, factor_details: Dict[str, Any]) -> tuple[str, str, str]:
        """
        Apply logical consistency checks to override verdicts when there are obvious contradictions.
        """
        # Check 1: If live verification says "no corroboration" but verdict is "real", override
        live_checker_data = factor_details.get('live_checker', {})
        live_decision = live_checker_data.get('decision', '').lower()
        
        if 'no corroboration' in live_decision and 'real' in verdict.lower():
            # Override to fake if no corroboration found
            if total_score >= 0.5:  # If score is already high enough
                return "LIKELY FAKE", "HIGH", f"Overridden: {reasoning} BUT no corroboration found in live news verification"
            else:
                # Adjust score and reasoning
                return "SUSPICIOUS", "HIGH", f"Overridden: {reasoning} BUT no corroboration found - requires verification"
        
        # Check 2: If BERT says "real" but other indicators strongly suggest fake, override
        bert_data = factor_details.get('bert', {})
        bert_decision = bert_data.get('decision', '')
        
        if bert_decision == 'REAL' and total_score >= 0.6:
            # High fake news score but BERT says real - suspicious
            return "SUSPICIOUS", "HIGH", f"Overridden: {reasoning} BUT high fake news indicators suggest verification needed"
        
        # Check 3: If claim density is very high but verdict is real, override UNLESS there's strong corroboration
        claim_data = factor_details.get('claim_density', {})
        semantic_density = claim_data.get('semantic_density', 0.0)
        
        if semantic_density > 0.9 and 'real' in verdict.lower():
            # Check if there's strong corroboration evidence
            if 'partial corroboration' in live_decision or 'strong corroboration' in live_decision:
                # If there's corroboration, don't override - the story might be real despite high claim density
                return verdict, confidence, f"{reasoning} - High claim density but corroboration evidence suggests this may be real news"
            else:
                # No corroboration + high claim density = suspicious
                return "SUSPICIOUS", "HIGH", f"Overridden: {reasoning} BUT extremely high claim density suggests verification needed"
        
        # Check 4: If there's strong corroboration evidence, don't override to fake
        if ('partial corroboration' in live_decision or 'strong corroboration' in live_decision) and 'fake' in verdict.lower():
            # Strong evidence suggests real news, don't override to fake
            return "LIKELY REAL", "HIGH", f"Overridden: {reasoning} BUT strong corroboration evidence suggests this is real news"
        
        # No overrides needed
        return verdict, confidence, reasoning
