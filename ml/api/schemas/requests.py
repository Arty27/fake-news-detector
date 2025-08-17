from pydantic import BaseModel, HttpUrl, validator
from typing import Optional

class TextAnalysisRequest(BaseModel):
    """Request model for text analysis"""
    text: str
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        if len(v.strip()) < 10:
            raise ValueError('Text must be at least 10 characters long')
        return v.strip()

class URLAnalysisRequest(BaseModel):
    """Request model for URL analysis"""
    url: HttpUrl
    
    @validator('url')
    def url_must_be_valid_news_url(cls, v):
        url_str = str(v)
        if not any(domain in url_str.lower() for domain in ['http', 'https']):
            raise ValueError('URL must be a valid HTTP/HTTPS URL')
        return v

class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis"""
    texts: list[str]
    
    @validator('texts')
    def texts_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        if len(v) > 10:
            raise ValueError('Maximum 10 texts allowed per batch')
        return v
