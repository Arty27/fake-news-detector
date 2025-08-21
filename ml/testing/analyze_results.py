import pandas as pd
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns


def map_verdict_to_binary(verdict: str) -> int:
    """Map verdict to binary (0 for real, 1 for fake) based on user mapping"""
    if pd.isna(verdict):
        return 0

    verdict_lower = str(verdict).upper().strip()

    # Mapping based on user specification
    fake_verdicts = ["DEFINITELY FAKE", "LIKELY FAKE", "SUSPICIOUS"]
    real_verdicts = ["LIKELY REAL", "DEFINITELY REAL"]

    if verdict_lower in fake_verdicts:
        return 1  # FAKE
    elif verdict_lower in real_verdicts:
        return 0  # REAL
    elif verdict_lower == "ERROR":
        print(
            f"Warning: Verdict '{verdict}' indicates an error - excluding from analysis"
        )
        return None  # Return None to exclude from analysis
    else:
        print(f"Warning: Unknown verdict '{verdict}' - treating as REAL")
        return 0  # Default to real if unclear


def map_true_label_to_binary(label: str) -> int:
    """Map true label to binary (0 for real, 1 for fake)"""
    if pd.isna(label):
        return 0

    label_lower = str(label).upper().strip()

    if label_lower == "FAKE":
        return 1
    elif label_lower == "REAL":
        return 0
    else:
        print(f"Warning: Unknown true label '{label}' - treating as REAL")
        return 0


def generate_roc_curve(true_labels, predictions, save_path="roc_curve.png"):
    """Generate and save AUC-ROC curve visualization"""
    try:
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        # Create ROC curve plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier (AUC = 0.500)')
        
        # Customize plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve\nFake News Detector Performance', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add AUC value annotation
        plt.text(0.6, 0.3, f'AUC = {roc_auc:.3f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=12, fontweight='bold')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curve saved to {save_path}")
        return roc_auc
        
    except Exception as e:
        print(f"Could not create ROC curve: {e}")
        return None


def load_and_analyze_results(filename: str = "batch_results.json"):
    """Load batch results and analyze them"""
    try:
        with open(filename, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results from {filename}")
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please run the batch script first.")
        return
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return

    if not results:
        print("No results found in the file.")
        return

    # Extract predictions and true labels
    predictions = []
    true_labels = []
    valid_results = []

    for result in results:
        # print(result["prediction"])

        # Check if we have a valid prediction
        if "prediction" in result and "final_verdict" in result["prediction"]:
            verdict = result["prediction"]["final_verdict"]
            true_label = result.get("true_label", "N/A")

            # Skip if no true label
            if true_label == "N/A":
                continue

            # Map to binary
            pred_binary = map_verdict_to_binary(verdict)
            true_binary = map_true_label_to_binary(true_label)

            # Skip if prediction mapping failed (e.g., ERROR verdicts)
            if pred_binary is None:
                continue

            predictions.append(pred_binary)
            true_labels.append(true_binary)
            valid_results.append(result)

    if not predictions:
        print("No valid predictions found for analysis.")
        return

    if len(predictions) < 2:
        print(f"\nWarning: Only {len(predictions)} valid prediction(s) found.")
        print("At least 2 predictions are needed for meaningful metrics.")
        print("Consider running more articles through the fake news detector.")
        return

    print(f"\nAnalyzing {len(predictions)} valid predictions...")

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    # Create confusion matrix with explicit labels to handle edge cases
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1])

    # Generate ROC curve and calculate AUC
    roc_auc = generate_roc_curve(true_labels, predictions)

    # Print results
    print("\n" + "=" * 60)
    print("FAKE NEWS DETECTOR PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Total Articles Analyzed: {len(predictions)}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    if roc_auc:
        print(f"AUC-ROC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")

    # Confusion Matrix
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print("                Predicted")
    print("                REAL  FAKE")

    # Handle edge cases where confusion matrix might be incomplete
    if cm.shape == (2, 2):
        print(f"Actual REAL    {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"      FAKE     {cm[1][0]:4d}  {cm[1][1]:4d}")
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle cases with only one class
        print("Note: Confusion matrix incomplete due to limited data")
        if len(set(true_labels)) == 1 and len(set(predictions)) == 1:
            if true_labels[0] == 0 and predictions[0] == 0:
                tn, fp, fn, tp = 1, 0, 0, 0  # Only REAL class
            elif true_labels[0] == 1 and predictions[0] == 1:
                tn, fp, fn, tp = 0, 0, 0, 1  # Only FAKE class
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

    print(f"\nTrue Negatives (TN): {tn} - Correctly identified REAL news")
    print(f"False Positives (FP): {fp} - REAL news incorrectly labeled as FAKE")
    print(f"False Negatives (FN): {fn} - FAKE news incorrectly labeled as REAL")
    print(f"True Positives (TP): {tp} - Correctly identified FAKE news")

    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2

    print(f"\nSpecificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"Sensitivity: {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f} ({balanced_accuracy*100:.2f}%)")

    # Detailed classification report
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    target_names = ["REAL", "FAKE"]
    report = classification_report(
        true_labels, predictions, target_names=target_names, digits=4
    )
    print(report)

    # Save detailed results
    detailed_results = {
        "summary": {
            "total_articles": len(predictions),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "sensitivity": sensitivity,
            "balanced_accuracy": balanced_accuracy,
            "auc_roc": roc_auc,
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "predictions": {"true_labels": true_labels, "predicted_labels": predictions},
    }

    with open("detailed_metrics.json", "w") as f:
        json.dump(detailed_results, f, indent=2, default=str)

    print(f"\nDetailed metrics saved to detailed_metrics.json")

    # Create and save confusion matrix visualization
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["REAL", "FAKE"],
            yticklabels=["REAL", "FAKE"],
        )
        plt.title("Confusion Matrix - Fake News Detector")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        print("Confusion matrix visualization saved to confusion_matrix.png")
    except Exception as e:
        print(f"Could not create visualization: {e}")

    return detailed_results


def main():
    print("Fake News Detector Results Analyzer")
    print("=" * 40)

    # Try to load batch results
    results = load_and_analyze_results("batch_results.json")

    if not results:
        print("\nTrying to load final batch results...")
        results = load_and_analyze_results("final_batch_results.json")

    if not results:
        print("\nNo results found. Please run the batch testing script first.")
        return

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
