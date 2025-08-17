import time
from datetime import datetime
from typing import Dict, Any, Optional
import json

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for API response"""
    return timestamp.isoformat()

def calculate_processing_time(start_time: float) -> int:
    """Calculate processing time in milliseconds"""
    return int((time.time() - start_time) * 1000)

def sanitize_text(text: str, max_length: int = 10000) -> str:
    """Sanitize and truncate text input"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text

def validate_url(url: str) -> bool:
    """Basic URL validation"""
    if not url:
        return False
    
    url_lower = url.lower()
    return url_lower.startswith(('http://', 'https://'))

def format_error_response(error: str, endpoint: str) -> Dict[str, Any]:
    """Format error response for API"""
    return {
        "success": False,
        "error": error,
        "endpoint": endpoint,
        "timestamp": datetime.now().isoformat()
    }

def format_success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    """Format success response for API"""
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }

def prepare_dashboard_data(fake_news_score: float) -> Dict[str, Any]:
    """Prepare standardized dashboard data"""
    if fake_news_score >= 0.75:
        color = '#dc3545'  # Red
        risk_level = 'high'
        verdict_category = 'definitely_fake'
    elif fake_news_score >= 0.6:
        color = '#fd7e14'  # Orange
        risk_level = 'high'
        verdict_category = 'likely_fake'
    elif fake_news_score >= 0.45:
        color = '#ffc107'  # Yellow
        risk_level = 'medium'
        verdict_category = 'suspicious'
    elif fake_news_score >= 0.3:
        color = '#20c997'  # Teal
        risk_level = 'medium'
        verdict_category = 'likely_real'
    else:
        color = '#198754'  # Green
        risk_level = 'low'
        verdict_category = 'definitely_real'
    
    return {
        'color': color,
        'risk_level': risk_level,
        'verdict_category': verdict_category,
        'summary': {
            'total_score': fake_news_score,
            'verdict_category': verdict_category,
            'risk_level': risk_level
        }
    }
