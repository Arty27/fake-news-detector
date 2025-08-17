#!/usr/bin/env python3
"""
Simple script to run the Fake News Detection API server
"""

import uvicorn

if __name__ == "__main__":
    print("Starting Fake News Detection API Server...")
    print("API Documentation will be available at: http://localhost:8000/docs")
    print("Health check available at: http://localhost:8000/health")
    print("=" * 60)

    uvicorn.run(
        "api.app:app",  # Use import string for reload to work
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info",
    )
