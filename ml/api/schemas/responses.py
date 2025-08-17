from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class FactorBreakdown(BaseModel):
    """Individual factor breakdown for dashboard"""
    score: float
    weight: float
    contribution: float
    decision: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class LiveNewsStory(BaseModel):
    """Individual news story from live verification"""
    title: str
    source: str
    url: str
    published_at: str
    similarity: float

class LiveNewsVerification(BaseModel):
    """Live news verification results"""
    verification_score: float
    decision: str
    queries_generated: int
    top_similarity: float
    stories_found: int
    search_queries: List[str]
    top_matching_stories: List[LiveNewsStory]

class NamedEntity(BaseModel):
    """Named entity information"""
    text: str
    label: str
    start: int
    end: int

class NamedEntitiesInfo(BaseModel):
    """Named entities analysis results"""
    entities_found: int
    entity_types: List[str]
    entities: List[NamedEntity]

class ClaimDensityInfo(BaseModel):
    """Claim density analysis results"""
    semantic_density: float
    claim_count: int
    sentence_count: int

class SentimentInfo(BaseModel):
    """Sentiment analysis results"""
    overall_sentiment: float
    sentiment_category: str
    negative_ratio: Optional[float] = None

class BERTAnalysis(BaseModel):
    """BERT classification results"""
    fake_probability: float
    real_probability: float
    decision: str
    confidence: float

class DashboardSummary(BaseModel):
    """Dashboard summary information"""
    color_code: str
    risk_level: str
    verdict_category: str
    total_score: float

class DashboardFactors(BaseModel):
    """Dashboard factors breakdown"""
    bert_classification: Dict[str, Any]
    live_verification: Dict[str, Any]
    claim_density: Dict[str, Any]
    named_entities: Dict[str, Any]
    sentiment: Dict[str, Any]

class AnalysisResult(BaseModel):
    """Complete analysis result for dashboard"""
    final_verdict: str
    confidence_level: str
    fake_news_score: float
    reasoning: str
    factor_breakdown: Dict[str, FactorBreakdown]
    dashboard_data: Dict[str, Any]
    analysis_timestamp: datetime
    processing_time_ms: int

class TextAnalysisResponse(BaseModel):
    """Response for text analysis endpoint"""
    success: bool
    message: str
    data: Optional[AnalysisResult] = None
    error: Optional[str] = None

class URLAnalysisResponse(BaseModel):
    """Response for URL analysis endpoint"""
    success: bool
    message: str
    data: Optional[AnalysisResult] = None
    error: Optional[str] = None
    url_info: Optional[Dict[str, Any]] = None

class BatchAnalysisResponse(BaseModel):
    """Response for batch analysis endpoint"""
    success: bool
    message: str
    total_processed: int
    successful_analyses: int
    failed_analyses: int
    results: List[AnalysisResult]
    processing_time_ms: int

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    models_loaded: bool
