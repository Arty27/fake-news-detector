# Fake News Detector Testing Scripts

This directory contains scripts to test the fake news detector on your CSV articles.

## Files

- `test_articles.py` - Simple script to test all articles at once
- `batch_test_articles.py` - Batch processing script that can resume from where it left off
- `requirements_test.txt` - Additional dependencies needed for testing

## Setup

1. Install the additional testing dependencies:
```bash
pip install -r requirements_test.txt
```

2. Make sure you have all the main dependencies installed:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Simple Testing (test_articles.py)

Run all articles at once:
```bash
python test_articles.py
```

This script:
- Loads your `news_articles.csv` file
- Combines title and content columns
- Removes the `...[...chars]` pattern from content
- Runs the comprehensive fake news detector on each article
- Saves results to `final_results.json`
- Calculates accuracy, precision, recall, and F1 score if labels are available
- Saves intermediate results every 10 articles

### Option 2: Batch Processing (batch_test_articles.py)

For handling API limits and resuming:
```bash
python batch_test_articles.py
```

This script:
- Saves results after each article to `batch_results.json`
- Can resume from where it left off if interrupted
- Useful when dealing with API rate limits
- Final results saved to `final_batch_results.json`

## CSV Format

Your CSV should have these columns:
- `title` - Article title
- `content` - Article content (will be cleaned of `...[...chars]` pattern)
- `label` - Ground truth label (optional, for calculating metrics)

## Output Files

- `final_results.json` / `final_batch_results.json` - Complete results for all articles
- `metrics.json` / `batch_metrics.json` - Accuracy metrics (if labels available)
- `intermediate_results_*.json` - Intermediate results during processing

## Handling API Limits

If you hit API limits:
1. Stop the script (Ctrl+C)
2. Wait for rate limit to reset
3. Run `batch_test_articles.py` again - it will resume from where it left off

## Notes

- The script combines title and content for analysis
- Content is cleaned by removing the `...[...chars]` pattern
- 1 second delay between articles to avoid overwhelming APIs
- Results are saved incrementally to prevent data loss
- If you get a new API key, just restart the batch script
