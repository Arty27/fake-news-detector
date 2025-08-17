# 🚨 Fake News Detection API

A comprehensive REST API for detecting fake news using multiple AI techniques. This API provides both text-based and URL-based analysis with dashboard-ready responses.

## 🏗️ **Architecture Overview**

```
ml/
├── api/                          # API package
│   ├── app.py                   # Main FastAPI application
│   ├── routes/                  # API endpoints
│   │   ├── text_analysis.py     # Text analysis endpoints
│   │   └── url_analysis.py      # URL analysis endpoints
│   ├── services/                # Business logic
│   │   ├── text_service.py      # Text analysis service
│   │   └── url_service.py       # URL analysis service
│   ├── schemas/                 # Request/Response models
│   │   ├── requests.py          # Input validation
│   │   └── responses.py         # Output formatting
│   └── utils/                   # Helper functions
│       └── helpers.py           # Utility functions
├── models/                       # Existing ML models (unchanged)
├── main.py                      # Cleaned up (no prints)
├── run_api.py                   # API startup script
└── requirements_api.txt          # API dependencies
```

## 🚀 **Quick Start**

### **1. Install API Dependencies**
```bash
pip install -r requirements_api.txt
```

### **2. Start the API Server**
```bash
python run_api.py
```

### **3. Access the API**
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Root Info**: http://localhost:8000/

## 📡 **API Endpoints**

### **Text Analysis**
- **`POST /text/analyze`** - Analyze single text article
- **`POST /text/analyze/batch`** - Analyze multiple text articles (up to 10)
- **`GET /text/health`** - Text analysis service health check

### **URL Analysis**
- **`POST /url/analyze`** - Analyze article from URL
- **`GET /url/health`** - URL analysis service health check

### **System Health**
- **`GET /health`** - Overall API health check
- **`GET /`** - API information and endpoint list

## 🔍 **Text Analysis Endpoint**

### **Request**
```json
POST /text/analyze
{
    "text": "Your article text here..."
}
```

### **Response**
```json
{
    "success": true,
    "message": "Text analysis completed successfully",
    "data": {
        "final_verdict": "LIKELY REAL",
        "confidence_level": "HIGH",
        "fake_news_score": 0.31,
        "reasoning": "Most indicators suggest this is likely real news",
        "factor_breakdown": {
            "bert": {
                "score": 0.000,
                "weight": 0.20,
                "contribution": 0.0000,
                "decision": "REAL"
            },
            "live_checker": {
                "score": 0.300,
                "weight": 0.40,
                "contribution": 0.1200,
                "decision": "partial corroboration"
            }
        },
        "dashboard_data": {
            "color": "#20c997",
            "risk_level": "medium",
            "verdict_category": "likely_real"
        },
        "analysis_timestamp": "2025-01-20T10:30:00",
        "processing_time_ms": 1250
    }
}
```

## 🌐 **URL Analysis Endpoint**

### **Request**
```json
POST /url/analyze
{
    "url": "https://example.com/article"
}
```

### **Response**
```json
{
    "success": true,
    "message": "URL analysis completed successfully",
    "data": {
        "final_verdict": "LIKELY REAL",
        "confidence_level": "HIGH",
        "fake_news_score": 0.25,
        "reasoning": "Analysis based on URL patterns, domain trustworthiness, and content analysis",
        "factor_breakdown": {...},
        "dashboard_data": {...},
        "analysis_timestamp": "2025-01-20T10:30:00",
        "processing_time_ms": 2100
    },
    "url_info": {
        "url": "https://example.com/article",
        "article_title": "Article Title",
        "article_authors": ["Author Name"],
        "article_publish_date": "2025-01-20",
        "domain_trustworthiness": {...},
        "live_news_similarity": {...}
    }
}
```

## 📊 **Dashboard-Ready Data**

All API responses include comprehensive data suitable for dashboard visualization:

### **Color Coding**
- 🟢 **Green** (`#198754`): Definitely Real (score < 0.3)
- 🟡 **Teal** (`#20c997`): Likely Real (score 0.3-0.45)
- 🟡 **Yellow** (`#ffc107`): Suspicious (score 0.45-0.6)
- 🟠 **Orange** (`#fd7e14`): Likely Fake (score 0.6-0.75)
- 🔴 **Red** (`#dc3545`): Definitely Fake (score > 0.75)

### **Risk Levels**
- **Low**: Definitely Real
- **Medium**: Likely Real, Suspicious
- **High**: Likely Fake, Definitely Fake

### **Factor Breakdown**
Each analysis includes detailed breakdown of:
- **BERT Classification**: AI-based fake news detection
- **Live News Verification**: Cross-referencing with NewsAPI
- **Claim Density**: Semantic similarity analysis
- **Named Entities**: Person/place/organization detection
- **Sentiment Analysis**: Emotional tone analysis

## 🛠️ **Development & Testing**

### **Run with Auto-reload**
```bash
python run_api.py
```

### **Run with Uvicorn directly**
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

### **Test Endpoints**
```bash
# Health check
curl http://localhost:8000/health

# Text analysis
curl -X POST "http://localhost:8000/text/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your test article text here..."}'
```

## 🔧 **Configuration**

### **Environment Variables**
Create a `.env` file in the `ml/` directory:
```env
NEWS_API_KEY=your_news_api_key_here
MODEL_CACHE_DIR=./models/cache
LOG_LEVEL=INFO
```

### **CORS Settings**
The API includes CORS middleware configured for development. For production, update the CORS settings in `api/app.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Production domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📈 **Performance & Scaling**

### **Model Loading**
- Models are loaded once when services initialize
- Subsequent requests reuse loaded models
- Memory usage optimized for production deployment

### **Batch Processing**
- Batch endpoint processes up to 10 texts simultaneously
- Useful for bulk analysis and comparison studies
- Processing time scales linearly with batch size

### **Error Handling**
- Comprehensive error handling with consistent response format
- Graceful degradation when individual models fail
- Detailed error messages for debugging

## 🚀 **Production Deployment**

### **Using Gunicorn**
```bash
gunicorn api.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### **Docker Support**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_api.txt .
RUN pip install -r requirements_api.txt
COPY . .
CMD ["python", "run_api.py"]
```

### **Load Balancing**
- API designed for horizontal scaling
- Stateless design allows multiple instances
- Health check endpoints for load balancer integration

## 🔒 **Security Considerations**

### **Input Validation**
- All inputs validated using Pydantic schemas
- Text length limits (10-10,000 characters)
- URL validation and sanitization
- Batch size limits (max 10 texts)

### **Rate Limiting**
Consider implementing rate limiting for production:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
```

## 📚 **API Documentation**

### **Interactive Docs**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### **Code Examples**
See the `examples/` directory for sample API usage in different programming languages.

## 🤝 **Contributing**

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** thoroughly
5. **Submit** a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 **Support**

- **Issues**: Create GitHub issues for bugs and feature requests
- **Documentation**: Check the API docs at `/docs` endpoint
- **Health Checks**: Use `/health` endpoints for system status

---

**🎯 Ready to detect fake news? Start the API and begin analyzing articles!**
