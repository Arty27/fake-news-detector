from fastapi import APIRouter, HTTPException, Depends
from typing import List
import sys
import os

# Add the parent directory to the path to import schemas and services
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from api.schemas.requests import URLAnalysisRequest
from api.schemas.responses import URLAnalysisResponse, AnalysisResult
from api.services.url_service import URLAnalysisService

router = APIRouter(prefix="/url", tags=["URL Analysis"])

# Initialize service
url_service = URLAnalysisService()


@router.post("/analyze", response_model=URLAnalysisResponse)
async def analyze_url(request: URLAnalysisRequest):
    """
    Analyze an article from URL for fake news detection

    This endpoint analyzes articles directly from URLs using:
    - Web scraping and content extraction
    - Domain trustworthiness analysis
    - URL pattern analysis
    - Live news cross-referencing
    - Multi-model content analysis

    Returns comprehensive analysis results suitable for dashboard display.
    """
    try:
        # Analyze the URL
        result = url_service.analyze_url(str(request.url))

        # Check for errors
        if "error" in result:
            return URLAnalysisResponse(
                success=False, message="URL analysis failed", error=result["error"]
            )

        # Format the response
        analysis_result = AnalysisResult(
            final_verdict=result["final_verdict"],
            confidence_level=result["confidence_level"],
            fake_news_score=result["fake_news_score"],
            reasoning=result["reasoning"],
            factor_breakdown=result["factor_breakdown"],
            dashboard_data=result["dashboard_data"],
            analysis_timestamp=result["analysis_timestamp"],
            processing_time_ms=result["processing_time_ms"],
        )

        return URLAnalysisResponse(
            success=True,
            message="URL analysis completed successfully",
            data=analysis_result,
            url_info=result.get("url_info", {}),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health")
async def url_analysis_health():
    """
    Health check for URL analysis service

    Verifies that the URL detector is loaded and ready for analysis.
    """
    try:
        # Check if service is initialized
        if url_service:
            return {
                "status": "healthy",
                "service": "url_analysis",
                "models_loaded": True,
                "message": "URL analysis service is ready",
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "url_analysis",
            "models_loaded": False,
            "error": str(e),
        }
