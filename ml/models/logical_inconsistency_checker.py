from typing import Dict, List, Tuple, Set
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class LogicalInconsistency:
    """Represents a logical inconsistency found in the text."""
    inconsistency_type: str
    description: str
    entities_involved: List[str]
    confidence: float
    rule_applied: str


class LogicalInconsistencyChecker:
    """
    Checks for logical inconsistencies in text by analyzing entity relationships
    and domain knowledge rules.
    """
    
    def __init__(self):
        # Initialize knowledge bases
        self._init_sports_knowledge()
        self._init_fictional_characters()
        self._init_domain_rules()
        logger.info("Logical Inconsistency Checker initialized")
    
    def _init_sports_knowledge(self):
        """Initialize sports-related knowledge."""
        self.sports_categories = {
            "cricket": ["cricket", "worldcup", "ipl", "test match", "odi", "t20"],
            "football": ["football", "fifa", "championship", "world cup", "premier league"],
            "basketball": ["basketball", "nba", "championship"],
            "tennis": ["tennis", "wimbledon", "us open", "australian open"],
            "olympics": ["olympics", "olympic games", "olympic"]
        }
        
        # Sports that cannot happen simultaneously or in same context
        self.incompatible_sports = {
            "cricket": ["football", "basketball", "tennis"],
            "football": ["cricket", "basketball", "tennis"],
            "basketball": ["cricket", "football", "tennis"],
            "tennis": ["cricket", "football", "basketball"]
        }
    
    def _init_fictional_characters(self):
        """Initialize fictional character database."""
        self.fictional_characters = {
            "harry potter": "fictional_wizard",
            "sherlock holmes": "fictional_detective",
            "spiderman": "fictional_superhero",
            "superman": "fictional_superhero",
            "batman": "fictional_superhero",
            "luke skywalker": "fictional_jedi",
            "gandalf": "fictional_wizard",
            "frodo": "fictional_hobbit",
            "iron man": "fictional_superhero",
            "captain america": "fictional_superhero"
        }
    
    def _init_domain_rules(self):
        """Initialize domain-specific logical rules."""
        self.domain_rules = {
            "fictional_participation": {
                "rule": "Fictional characters cannot participate in real-world events",
                "confidence": 0.95
            },
            "sport_mismatch": {
                "rule": "Different sports events cannot happen simultaneously in same context",
                "confidence": 0.90
            },
            "temporal_impossibility": {
                "rule": "Events with conflicting timeframes cannot occur together",
                "confidence": 0.85
            }
        }
    
    def check(self, text: str) -> Dict[str, any]:
        """
        Check text for logical inconsistencies.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing inconsistency results and overall score
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string")
        
        text_lower = text.lower()
        
        # Extract entities and check for inconsistencies
        inconsistencies = []
        
        # Check for fictional character participation in real events
        fictional_inconsistencies = self._check_fictional_character_participation(text_lower)
        inconsistencies.extend(fictional_inconsistencies)
        
        # Check for sport mismatches
        sport_inconsistencies = self._check_sport_mismatches(text_lower)
        inconsistencies.extend(sport_inconsistencies)
        
        # Check for temporal impossibilities
        temporal_inconsistencies = self._check_temporal_impossibilities(text_lower)
        inconsistencies.extend(temporal_inconsistencies)
        
        # Calculate overall inconsistency score
        overall_score = self._calculate_inconsistency_score(inconsistencies)
        
        return {
            "inconsistencies": inconsistencies,
            "inconsistency_count": len(inconsistencies),
            "overall_inconsistency_score": overall_score,
            "is_logically_consistent": overall_score < 0.3,
            "confidence": self._calculate_confidence(inconsistencies)
        }
    
    def _check_fictional_character_participation(self, text: str) -> List[LogicalInconsistency]:
        """Check if fictional characters are participating in real events."""
        inconsistencies = []
        
        for character, character_type in self.fictional_characters.items():
            if character in text:
                # Check if they're participating in sports or real events
                for sport_category, sport_terms in self.sports_categories.items():
                    for sport_term in sport_terms:
                        if sport_term in text:
                            # Check if character is actively participating
                            participation_verbs = ["won", "played", "participated", "competed", "scored", "hit"]
                            for verb in participation_verbs:
                                if verb in text:
                                    inconsistencies.append(LogicalInconsistency(
                                        inconsistency_type="fictional_participation",
                                        description=f"Fictional character '{character}' cannot participate in real {sport_category} event",
                                        entities_involved=[character, sport_term],
                                        confidence=0.95,
                                        rule_applied="Fictional characters cannot participate in real-world events"
                                    ))
                                    break
                            break
                    if any(sport_term in text for sport_term in sport_terms):
                        break
        
        return inconsistencies
    
    def _check_sport_mismatches(self, text: str) -> List[LogicalInconsistency]:
        """Check for incompatible sports combinations."""
        inconsistencies = []
        
        # Find all sports mentioned in the text
        mentioned_sports = []
        for sport_category, sport_terms in self.sports_categories.items():
            for sport_term in sport_terms:
                if sport_term in text:
                    mentioned_sports.append(sport_category)
                    break
        
        # Check for incompatible combinations
        if len(mentioned_sports) > 1:
            for i, sport1 in enumerate(mentioned_sports):
                for sport2 in mentioned_sports[i+1:]:
                    if sport2 in self.incompatible_sports.get(sport1, []):
                        inconsistencies.append(LogicalInconsistency(
                            inconsistency_type="sport_mismatch",
                            description=f"Incompatible sports combination: {sport1} and {sport2} cannot occur in same context",
                            entities_involved=[sport1, sport2],
                            confidence=0.90,
                            rule_applied="Different sports events cannot happen simultaneously in same context"
                        ))
        
        return inconsistencies
    
    def _check_temporal_impossibilities(self, text: str) -> List[LogicalInconsistency]:
        """Check for temporal impossibilities."""
        inconsistencies = []
        
        # Check for impossible time combinations
        time_indicators = {
            "past_events": ["ancient", "medieval", "1920s", "1930s", "1940s", "1950s", "1960s", "1970s", "1980s", "1990s"],
            "future_events": ["2025", "2026", "2027", "2028", "2029", "2030", "future", "next year", "next decade"],
            "current_events": ["2024", "this year", "current", "present", "now"]
        }
        
        mentioned_times = []
        for time_category, time_terms in time_indicators.items():
            for time_term in time_terms:
                if time_term in text:
                    mentioned_times.append(time_category)
                    break
        
        # Check for conflicting time categories
        if len(mentioned_times) > 1:
            if "past_events" in mentioned_times and "future_events" in mentioned_times:
                inconsistencies.append(LogicalInconsistency(
                    inconsistency_type="temporal_impossibility",
                    description="Past and future events cannot occur simultaneously",
                    entities_involved=["past_events", "future_events"],
                    confidence=0.85,
                    rule_applied="Events with conflicting timeframes cannot occur together"
                ))
        
        return inconsistencies
    
    def _calculate_inconsistency_score(self, inconsistencies: List[LogicalInconsistency]) -> float:
        """Calculate overall inconsistency score (0.0 = consistent, 1.0 = highly inconsistent)."""
        if not inconsistencies:
            return 0.0
        
        # Weight inconsistencies by confidence and type
        total_score = 0.0
        total_weight = 0.0
        
        for inconsistency in inconsistencies:
            weight = 1.0
            if inconsistency.inconsistency_type == "fictional_participation":
                weight = 1.5  # Higher weight for obvious impossibilities
            elif inconsistency.inconsistency_type == "sport_mismatch":
                weight = 1.2  # Medium weight for sport conflicts
            
            total_score += inconsistency.confidence * weight
            total_weight += weight
        
        return min(1.0, total_score / total_weight)
    
    def _calculate_confidence(self, inconsistencies: List[LogicalInconsistency]) -> float:
        """Calculate overall confidence in the analysis."""
        if not inconsistencies:
            return 1.0
        
        # Average confidence of all detected inconsistencies
        avg_confidence = sum(inc.confidence for inc in inconsistencies) / len(inconsistencies)
        return avg_confidence

