import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def analyze_existing_results(filename='batch_results.json'):
    """Analyze existing batch results from JSON file"""
    
    # Load the results
    try:
        with open(filename, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results from {filename}")
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return
    
    if not results:
        print("No results found.")
        return
    
    # Examine the first few results to understand structure
    print("\n" + "="*60)
    print("EXAMINING RESULTS STRUCTURE")
    print("="*60)
    
    if results:
        first_result = results[0]
        print("Sample result structure:")
        for key, value in first_result.items():
            if key == 'prediction' and isinstance(value, dict):
                print(f"  {key}:")
                for pred_key, pred_value in value.items():
                    print(f"    {pred_key}: {pred_value}")
            else:
                print(f"  {key}: {value}")
    
    # Count different verdict types
    verdict_counts = {}
    error_details = []
    
    for result in results:
        if 'prediction' in result and 'verdict' in result['prediction']:
            verdict = result['prediction']['verdict']
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            
            # Collect error details
            if verdict == 'ERROR' and 'error' in result['prediction']:
                error_details.append(result['prediction']['error'])
    
    print("\n" + "="*60)
    print("VERDICT DISTRIBUTION")
    print("="*60)
    for verdict, count in verdict_counts.items():
        print(f"{verdict}: {count}")
    
    if error_details:
        print(f"\nERROR DETAILS (showing first 5):")
        for i, error in enumerate(error_details[:5]):
            print(f"  {i+1}. {error}")
    
    # Check if we have any valid predictions for analysis
    valid_predictions = []
    valid_true_labels = []
    
    for result in results:
        if ('prediction' in result and 
            'verdict' in result['prediction'] and 
            result['prediction']['verdict'] != 'ERROR' and
            result.get('true_label', 'N/A') != 'N/A'):
            
            verdict = result['prediction']['verdict']
            true_label = result['true_label']
            
            # Map to binary
            if verdict in ['DEFINITELY FAKE', 'LIKELY FAKE', 'SUSPICIOUS']:
                pred_binary = 1  # FAKE
            elif verdict in ['LIKELY REAL', 'DEFINITELY REAL']:
                pred_binary = 0  # REAL
            else:
                continue  # Skip unknown verdicts
            
            if true_label.upper() == 'FAKE':
                true_binary = 1
            elif true_label.upper() == 'REAL':
                true_binary = 0
            else:
                continue  # Skip unknown labels
            
            valid_predictions.append(pred_binary)
            valid_true_labels.append(true_binary)
    
    print(f"\n" + "="*60)
    print("VALID PREDICTIONS FOR ANALYSIS")
    print("="*60)
    print(f"Found {len(valid_predictions)} valid predictions out of {len(results)} total results")
    
    if len(valid_predictions) >= 2:
        print("\nCalculating metrics...")
        
        # Calculate metrics
        accuracy = accuracy_score(valid_true_labels, valid_predictions)
        precision = precision_score(valid_true_labels, valid_predictions, zero_division=0)
        recall = recall_score(valid_true_labels, valid_predictions, zero_division=0)
        f1 = f1_score(valid_true_labels, valid_predictions, zero_division=0)
        
        print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
        
        # Confusion Matrix
        cm = confusion_matrix(valid_true_labels, valid_predictions, labels=[0, 1])
        print(f"\nConfusion Matrix:")
        print("                Predicted")
        print("                REAL  FAKE")
        print(f"Actual REAL    {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"      FAKE     {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2
        
        print(f"\nSpecificity: {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"Sensitivity: {sensitivity:.4f} ({sensitivity*100:.2f}%)")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f} ({balanced_accuracy*100:.2f}%)")
        
        # Save results
        analysis_results = {
            "summary": {
                "total_results": len(results),
                "valid_predictions": len(valid_predictions),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "specificity": specificity,
                "sensitivity": sensitivity,
                "balanced_accuracy": balanced_accuracy
            },
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            },
            "verdict_distribution": verdict_counts
        }
        
        with open('analysis_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        print(f"\nAnalysis results saved to analysis_results.json")
        
    else:
        print(f"\nInsufficient valid predictions for analysis.")
        print(f"Need at least 2 valid predictions, found {len(valid_predictions)}")
        
        if len(valid_predictions) == 1:
            print(f"\nSingle prediction details:")
            print(f"Predicted: {'FAKE' if valid_predictions[0] == 1 else 'REAL'}")
            print(f"True Label: {'FAKE' if valid_true_labels[0] == 1 else 'REAL'}")
            print(f"Result: {'CORRECT' if valid_predictions[0] == valid_true_labels[0] else 'INCORRECT'}")

if __name__ == "__main__":
    print("Existing Results Analyzer")
    print("="*40)
    analyze_existing_results()
