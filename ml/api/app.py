from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import sys
import os

# Add the parent directory to the path to import routes
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from api.routes.text_analysis import router as text_router
from api.routes.url_analysis import router as url_router

# Create FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="""
    A comprehensive API for detecting fake news using multiple AI techniques.
    
    ## Features
    
    * **Text Analysis**: Analyze article text using BERT, sentiment analysis, NER, claim density, and live news verification
    * **URL Analysis**: Analyze articles directly from URLs with web scraping and domain analysis
    * **Batch Processing**: Process multiple articles simultaneously
    * **Dashboard Ready**: All responses include comprehensive data suitable for dashboard visualization
    
    ## Models Used
    
    * BERT Classification for fake news detection
    * Sentiment Analysis for emotional tone
    * Named Entity Recognition for person/place/organization detection
    * Claim Density Analysis for semantic similarity
    * Live News Verification using NewsAPI
    * Domain Trustworthiness Analysis
    * URL Pattern Analysis
    
    ## Endpoints
    
    * `POST /text/analyze` - Analyze single text article
    * `POST /text/analyze/batch` - Analyze multiple text articles
    * `POST /url/analyze` - Analyze article from URL
    * `GET /health` - Overall API health check
    * `GET /text/health` - Text analysis service health
    * `GET /url/health` - URL analysis service health
    """,
    version="1.0.0",
    contact={
        "name": "Fake News Detection System",
        "url": "https://github.com/your-repo/fake-news-detection",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(text_router)
app.include_router(url_router)

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint providing API information
    """
    return {
        "message": "Fake News Detection API",
        "version": "1.0.0",
        "description": "Comprehensive fake news detection using multiple AI techniques",
        "endpoints": {
            "text_analysis": "/text/analyze",
            "batch_text_analysis": "/text/analyze/batch",
            "url_analysis": "/url/analyze",
            "health_check": "/health",
            "documentation": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Overall API health check
    
    Returns the status of all services and the overall API health.
    """
    try:
        # Check text analysis service
        from api.services.text_service import TextAnalysisService
        text_service_healthy = True
        try:
            text_service = TextAnalysisService()
        except:
            text_service_healthy = False
        
        # Check URL analysis service
        from api.services.url_service import URLAnalysisService
        url_service_healthy = True
        try:
            url_service = URLAnalysisService()
        except:
            url_service_healthy = False
        
        overall_healthy = text_service_healthy and url_service_healthy
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "services": {
                "text_analysis": {
                    "status": "healthy" if text_service_healthy else "unhealthy",
                    "models_loaded": text_service_healthy
                },
                "url_analysis": {
                    "status": "healthy" if url_service_healthy else "unhealthy",
                    "models_loaded": url_service_healthy
                }
            },
            "overall_health": overall_healthy
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "error": str(e),
            "overall_health": False
        }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with consistent error format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions with consistent error format"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "details": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
