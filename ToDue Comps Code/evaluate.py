#!/usr/bin/env python3
"""
Evaluate the assignment extraction system.

To reproduce the paper results, you would need the test set of 30 annotated syllabi .
"""

from syllabus_parser import SyllabusParser
import json
import pickle
from collections import defaultdict


def calculate_metrics(predictions, ground_truth):
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        predictions: List of predicted assignments
        ground_truth: List of ground truth assignments
        
    Returns:
        Dictionary with TP, FP, FN, precision, recall, F1
    """
    # Convert to sets for comparison (simplified - real implementation uses fuzzy matching)
    pred_set = set(predictions)
    truth_set = set(ground_truth)
    
    tp = len(pred_set & truth_set)  # True positives
    fp = len(pred_set - truth_set)  # False positives
    fn = len(truth_set - pred_set)  # False negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_by_format(test_data):
    """
    Evaluate performance by syllabus format (table, bullet, prose).
    
    Args:
        test_data: Dictionary mapping format -> list of test cases
        
    Returns:
        Dictionary of metrics by format
    """
    results = {}
    
    for format_type, test_cases in test_data.items():
        format_results = {
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
            'syllabi_count': len(test_cases)
        }
        
        for test_case in test_cases:
            metrics = calculate_metrics(
                test_case['predicted'],
                test_case['ground_truth']
            )
            format_results['total_tp'] += metrics['tp']
            format_results['total_fp'] += metrics['fp']
            format_results['total_fn'] += metrics['fn']
        
        # Calculate aggregate metrics
        tp = format_results['total_tp']
        fp = format_results['total_fp']
        fn = format_results['total_fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        format_results['precision'] = precision
        format_results['recall'] = recall
        format_results['f1'] = f1
        
        results[format_type] = format_results
    
    return results


def load_test_data(test_data_path):
    """
    Load test data from JSON file.
    
    Expected format:
    {
        "tables": [
            {
                "syllabus": "course1.pdf",
                "predicted": ["Assignment 1", "Midterm"],
                "ground_truth": ["Assignment 1", "Midterm", "Final"]
            },
            ...
        ],
        "bullets": [...],
        "prose": [...]
    }
    """
    try:
        with open(test_data_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Test data not found at {test_data_path}")
        print("\nTo evaluate, you need a test set with ground truth annotations.")
        return None


def run_evaluation(test_data_path='test_data.json'):
    """
    Run full evaluation on test set.
    """
    print("Assignment Extraction Evaluation")
    print("=" * 60)
    print()
    
    # Load test data
    test_data = load_test_data(test_data_path)
    
    if not test_data:
        print("\nExample test data format:")
        example = {
            "tables": [
                {
                    "syllabus": "course1.pdf",
                    "predicted": ["Assignment 1 due Oct 15", "Midterm Oct 30"],
                    "ground_truth": ["Assignment 1 due Oct 15", "Midterm Oct 30", "Final Dec 10"]
                }
            ],
            "bullets": [],
            "prose": []
        }
        print(json.dumps(example, indent=2))
        return
    
    # Evaluate by format
    results = evaluate_by_format(test_data)
    
    # Print results
    print("\nResults by Format:")
    print("-" * 60)
    
    for format_type, metrics in results.items():
        print(f"\n{format_type.upper()}:")
        print(f"  Syllabi: {metrics['syllabi_count']}")
        print(f"  TP: {metrics['total_tp']}, FP: {metrics['total_fp']}, FN: {metrics['total_fn']}")
        print(f"  Precision: {metrics['precision']:.1%}")
        print(f"  Recall: {metrics['recall']:.1%}")
        print(f"  F1 Score: {metrics['f1']:.1%}")
    
    # Calculate overall metrics
    total_tp = sum(r['total_tp'] for r in results.values())
    total_fp = sum(r['total_fp'] for r in results.values())
    total_fn = sum(r['total_fn'] for r in results.values())
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE:")
    print(f"  Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"  Precision: {overall_precision:.1%}")
    print(f"  Recall: {overall_recall:.1%}")
    print(f"  F1 Score: {overall_f1:.1%}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        test_data_path = sys.argv[1]
    else:
        test_data_path = 'test_data.json'
        print(f"Usage: python evaluate.py [test_data.json]")
        print(f"Using default: {test_data_path}\n")
    
    run_evaluation(test_data_path)
