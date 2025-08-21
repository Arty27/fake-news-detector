import pandas as pd
import json
import time
import os
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
from models.bert_classifier import BERTFakeNewsClassifier
from models.sentiment_analysis import SentimentAnalyzer
from models.named_entitiy_recognition import NERExtractor
from models.claim_density import ClaimDensityScore
from models.live_checker import LiveCheckerService
from models.comprehensive_fake_news_detector import CombinedFakeNewsDetector

def clean_text(text: str) -> str:
    """Clean text by removing ...[...chars] pattern and extra whitespace"""
    if pd.isna(text):
        return ""
    
    # Remove the ...[...chars] pattern
    text = re.sub(r'\.\.\.\[.*?chars\]', '', str(text))
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def combine_title_content(title: str, content: str) -> str:
    """Combine title and content with proper formatting"""
    title_clean = clean_text(title)
    content_clean = clean_text(content)
    
    if title_clean and content_clean:
        return f"{title_clean}. {content_clean}"
    elif title_clean:
        return title_clean
    elif content_clean:
        return content_clean
    else:
        return ""

def predict_fake_news(text: str) -> Dict:
    """Run the comprehensive fake news detector on a text"""
    try:
        # Initialize models
        classifier = BERTFakeNewsClassifier()
        analyzer = SentimentAnalyzer()
        ner = NERExtractor()
        scorer = ClaimDensityScore()
        live_checker = LiveCheckerService()
        comprehensive_detector = CombinedFakeNewsDetector()

        # Run all analyses
        p_fake, p_real = classifier.predict(text)
        sentiment_result = analyzer.analyze(text)
        entities = ner.extract(text)
        claim_result = scorer.score(text)
        live_result = live_checker.check(text)

        # Prepare data for comprehensive detection
        bert_data = {"fake_probability": p_fake, "real_probability": p_real}

        # Run comprehensive detection
        final_result = comprehensive_detector.detect(
            bert_data, sentiment_result, entities, claim_result, live_result
        )

        return final_result
    except Exception as e:
        return {"error": str(e), "verdict": "ERROR"}

def convert_verdict_to_binary(verdict: str) -> int:
    """Convert verdict to binary (0 for real, 1 for fake)"""
    if pd.isna(verdict):
        return 0
    
    verdict_lower = str(verdict).lower()
    
    if any(word in verdict_lower for word in ['fake', 'suspicious']):
        return 1
    elif any(word in verdict_lower for word in ['real']):
        return 0
    else:
        return 0  # Default to real if unclear

def load_existing_results(filename: str) -> List[Dict]:
    """Load existing results from a JSON file"""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_results(results: List[Dict], filename: str):
    """Save results to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """Calculate accuracy, precision, recall, and F1 score"""
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    print("Loading CSV file...")
    
    # Load the CSV file with different encoding attempts
    df = None
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            print(f"Trying to load CSV with {encoding} encoding...")
            df = pd.read_csv('news_articles.csv', encoding=encoding)
            print(f"Successfully loaded {len(df)} articles with {encoding} encoding")
            break
        except Exception as e:
            print(f"Failed with {encoding}: {e}")
            continue
    
    if df is None:
        print("Error: Could not load CSV with any encoding. Please check the file.")
        return
    
    # Check if required columns exist
    if 'title' not in df.columns or 'content' not in df.columns:
        print("Error: CSV must contain 'title' and 'content' columns")
        return
    
    # Check if label column exists for evaluation
    has_labels = 'label' in df.columns
    
    # Load existing results if any
    results = load_existing_results('batch_results.json')
    processed_indices = {r['index'] for r in results}
    
    print(f"Found {len(results)} existing results")
    print(f"Already processed indices: {sorted(processed_indices)}")
    
    # Continue from where we left off
    start_idx = max(processed_indices) + 1 if processed_indices else 0
    print(f"Starting from index: {start_idx}")
    
    predictions = []
    true_labels = []
    
    # Add existing predictions to our lists
    for result in results:
        if 'prediction' in result and 'verdict' in result['prediction']:
            pred_binary = convert_verdict_to_binary(result['prediction']['verdict'])
            predictions.append(pred_binary)
            
            if has_labels and result['true_label'] != 'N/A':
                true_binary = convert_verdict_to_binary(result['true_label'])
                true_labels.append(true_binary)
    
    print(f"Loaded {len(predictions)} existing predictions")
    
    print("\nStarting fake news detection...")
    print("=" * 50)
    
    for idx in range(start_idx, len(df)):
        print(f"Processing article {idx + 1}/{len(df)}")
        
        # Combine title and content
        combined_text = combine_title_content(df.iloc[idx]['title'], df.iloc[idx]['content'])
        
        if not combined_text:
            print(f"  Skipping article {idx + 1} - no valid text")
            continue
        
        # Run fake news detection
        result = predict_fake_news(combined_text)
        
        # Store results
        article_result = {
            "index": idx,
            "title": df.iloc[idx]['title'],
            "text_length": len(combined_text),
            "prediction": result,
            "true_label": df.iloc[idx].get('label', 'N/A') if has_labels else 'N/A'
        }
        
        results.append(article_result)
        
        # Extract prediction for metrics
        if 'verdict' in result:
            pred_binary = convert_verdict_to_binary(result['verdict'])
            predictions.append(pred_binary)
            
            if has_labels:
                true_binary = convert_verdict_to_binary(df.iloc[idx]['label'])
                true_labels.append(true_binary)
        
        # Save results after each article
        save_results(results, 'batch_results.json')
        
        # Add delay to avoid API rate limits
        time.sleep(1)
        
        # Print progress every 5 articles
        if (idx + 1) % 5 == 0:
            print(f"  Progress: {idx + 1}/{len(df)} articles processed")
            print(f"  Current results saved to batch_results.json")
    
    print("\n" + "=" * 50)
    print("Analysis complete!")
    
    # Save final results
    save_results(results, 'final_batch_results.json')
    print(f"Final results saved to final_batch_results.json")
    
    # Calculate metrics if labels are available
    if has_labels and len(predictions) == len(true_labels) and len(predictions) > 0:
        print("\nCalculating metrics...")
        metrics = calculate_metrics(true_labels, predictions)
        
        if "error" not in metrics:
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            
            # Save metrics
            with open('batch_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            print("Metrics saved to batch_metrics.json")
        else:
            print(f"Error calculating metrics: {metrics['error']}")
    else:
        print("\nNo labels available for metric calculation")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total articles processed: {len(results)}")
    print(f"Successful predictions: {len(predictions)}")
    
    if has_labels:
        print(f"Articles with labels: {len(true_labels)}")

if __name__ == "__main__":
    main()
