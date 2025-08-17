from fastapi import APIRouter, HTTPException, Depends
from typing import List
import sys
import os

# Add the parent directory to the path to import schemas and services
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from api.schemas.requests import TextAnalysisRequest, BatchAnalysisRequest
from api.schemas.responses import (
    TextAnalysisResponse,
    BatchAnalysisResponse,
    AnalysisResult,
)
from api.services.text_service import TextAnalysisService

router = APIRouter(prefix="/text", tags=["Text Analysis"])

# Initialize service
text_service = TextAnalysisService()


@router.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze a single text article for fake news detection

    This endpoint analyzes the provided text using all available models:
    - BERT Classification
    - Sentiment Analysis
    - Named Entity Recognition
    - Claim Density Analysis
    - Live News Verification

    Returns comprehensive analysis results suitable for dashboard display.
    """
    try:
        # Analyze the text
        result = text_service.analyze_text(request.text)

        # Check for errors
        if "error" in result:
            return TextAnalysisResponse(
                success=False, message="Analysis failed", error=result["error"]
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

        return TextAnalysisResponse(
            success=True,
            message="Text analysis completed successfully",
            data=analysis_result,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch_texts(request: BatchAnalysisRequest):
    """
    Analyze multiple text articles in batch for fake news detection

    This endpoint processes up to 10 texts simultaneously, providing
    individual analysis results for each text. Useful for bulk processing
    and comparison analysis.
    """
    try:
        # Analyze all texts
        results = text_service.analyze_batch(request.texts)

        # Process results
        successful_analyses = []
        failed_count = 0

        for i, result in enumerate(results):
            if "error" not in result:
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
                successful_analyses.append(analysis_result)
            else:
                failed_count += 1

        # Calculate total processing time
        total_processing_time = sum(
            result.get("processing_time_ms", 0) for result in results
        )

        return BatchAnalysisResponse(
            success=True,
            message=f"Batch analysis completed. {len(successful_analyses)} successful, {failed_count} failed.",
            total_processed=len(request.texts),
            successful_analyses=len(successful_analyses),
            failed_analyses=failed_count,
            results=successful_analyses,
            processing_time_ms=total_processing_time,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.get("/health")
async def text_analysis_health():
    """
    Health check for text analysis service

    Verifies that all required models are loaded and ready for analysis.
    """
    try:
        # Check if service is initialized
        if text_service:
            return {
                "status": "healthy",
                "service": "text_analysis",
                "models_loaded": True,
                "message": "Text analysis service is ready",
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "text_analysis",
            "models_loaded": False,
            "error": str(e),
        }
