# 🎯 Unified Fake News Detection System

A comprehensive fake news detection system that combines **text-based analysis** and **URL-based analysis** using multiple AI techniques.

## 🚀 Features

### **Text-Based Analysis** (Original System)

- 🤖 **BERT Classification** - Deep learning-based fake news detection
- 😊 **Sentiment Analysis** - Emotional tone analysis
- 🏷️ **Named Entity Recognition** - Person, place, organization detection
- 📊 **Claim Density Analysis** - Semantic similarity and claim patterns
- 🌐 **Live News Verification** - Cross-reference with real news sources
- 🎯 **Comprehensive Scoring** - Weighted combination of all factors

### **URL-Based Analysis** (New System)

- 🌐 **Web Scraping** - Extract articles directly from URLs
- 🔗 **Domain Trustworthiness** - Check source credibility
- 📰 **Article Metadata** - Title, authors, publication date
- 🌐 **Live News Cross-Reference** - Verify stories against real news
- 🚨 **URL Pattern Analysis** - Detect suspicious URL structures
- 📊 **Multi-Model Analysis** - BERT, sentiment, NER, claim density

## 🛠️ Installation

### **1. Install Dependencies**

```bash
# For text-based analysis (existing)
pip install -r requirements.txt

# For URL-based analysis (new)
pip install -r requirements_url_detector.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

### **2. Environment Setup**

```bash
# Set your News API key (optional, for live verification)
export NEWS_API_KEY="your_api_key_here"
```

## 🎮 Usage

### **Interactive Mode (Recommended)**

```bash
python main.py
```

Choose from:

1. **Analyze text article** - Paste or type article text
2. **Analyze article from URL** - Provide article URL
3. **Run demo examples** - See both systems in action

### **Direct Function Calls**

```python
from main import analyze_text_article, analyze_url_article

# Analyze text
text = "Your article text here..."
analyze_text_article(text)

# Analyze URL
url = "https://example.com/article"
analyze_url_article(url)
```

## 📊 Understanding Results

### **Text Analysis Results**

- 🚨 **Final Verdict**: DEFINITELY FAKE, LIKELY FAKE, SUSPICIOUS, LIKELY REAL, DEFINITELY REAL
- 🎯 **Confidence Level**: VERY HIGH, HIGH, MEDIUM
- 📊 **Fake News Score**: 0.0 (real) to 1.0 (fake)
- 💡 **Reasoning**: Clear explanation of the decision
- 📋 **Factor Breakdown**: How each component contributed

### **URL Analysis Results**

- 🌐 **URL & Domain**: Source information
- 📰 **Article Metadata**: Title, authors, publication date
- 🚨 **Prediction**: likely_real, uncertain, likely_fake
- 🔗 **Domain Analysis**: Trustworthiness assessment
- 🌐 **Live Verification**: Cross-reference results
- 📊 **Detailed Scores**: BERT, sentiment, claim density

## 🔧 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UNIFIED SYSTEM                          │
├─────────────────────────────────────────────────────────────┤
│  📝 TEXT INPUT                    🌐 URL INPUT            │
│       │                                │                   │
│       ▼                                ▼                   │
│  ┌─────────────────┐            ┌─────────────────┐       │
│  │ Text Analyzer   │            │ URL Analyzer    │       │
│  │                 │            │                 │       │
│  │ • BERT          │            │ • Web Scraping  │       │
│  │ • Sentiment     │            │ • Domain Check  │       │
│  │ • NER           │            │ • Live Verify   │       │
│  │ • Claim Density │            │ • Pattern Check │       │
│  │ • Live Check    │            │ • Multi-Model   │       │
│  └─────────────────┘            └─────────────────┘       │
│       │                                │                   │
│       └────────────────┬───────────────┘                   │
│                        ▼                                   │
│              ┌─────────────────┐                           │
│              │ Unified Output  │                           │
│              │                 │                           │
│              │ • Final Verdict │                           │
│              │ • Confidence    │                           │
│              │ • Score         │                           │
│              │ • Reasoning     │                           │
│              │ • Dashboard     │                           │
│              └─────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## 🧪 Testing

### **Run Integration Tests**

```bash
python test_integration.py
```

### **Test Individual Components**

```python
# Test text detector
from models.comprehensive_fake_news_detector import CombinedFakeNewsDetector
detector = CombinedFakeNewsDetector()

# Test URL detector
from models.url_fake_news_detector import URLFakeNewsDetector
url_detector = URLFakeNewsDetector()
```

## 📈 Performance & Accuracy

### **Text Analysis**

- **Speed**: Fast (local processing)
- **Accuracy**: High (trained models)
- **Coverage**: Comprehensive (5 analysis factors)
- **Use Case**: Direct text input, batch processing

### **URL Analysis**

- **Speed**: Medium (web scraping + analysis)
- **Accuracy**: High (multiple verification layers)
- **Coverage**: Extensive (URL + content + cross-reference)
- **Use Case**: Online articles, source verification

## 🚨 Troubleshooting

### **Common Issues**

1. **Import Errors**

   ```bash
   # Install missing packages
   pip install -r requirements_url_detector.txt
   ```

2. **spaCy Model Missing**

   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **URL Access Issues**

   - Check internet connection
   - Verify URL is accessible
   - Some sites block automated access

4. **Model Loading Errors**
   - Ensure sufficient RAM (4GB+ recommended)
   - Check GPU drivers if using CUDA

### **Performance Tips**

- Use GPU if available for faster BERT processing
- Limit text length for very long articles
- Cache results for repeated analysis

## 🔮 Future Enhancements

- **Multi-language Support** - Non-English articles
- **Real-time Monitoring** - Continuous URL monitoring
- **API Endpoints** - REST API for integration
- **Batch Processing** - Multiple articles at once
- **Custom Models** - User-trained classifiers
- **Visual Dashboard** - Web-based interface

## 📚 API Reference

### **CombinedFakeNewsDetector**

```python
detector = CombinedFakeNewsDetector()
result = detector.detect(bert_data, sentiment, entities, claims, live_check)
```

### **URLFakeNewsDetector**

```python
detector = URLFakeNewsDetector()
result = detector.analyze_url("https://example.com/article")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🎉 Ready to detect fake news like a pro!**

Run `python main.py` to get started with the interactive system.
