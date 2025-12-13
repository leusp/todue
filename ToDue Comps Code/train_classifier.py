#!/usr/bin/env python3
"""
Train the assignment classifier on annotated syllabi.

This demonstrates the training process. To reproduce the results from
the paper, you would need:
- 30 annotated syllabi (12 tables, 11 bullets, 7 prose formats)
- Institutional data (not included for privacy)
"""

from syllabus_parser import SyllabusParser
import pickle
import json


def load_training_data(ground_truth_path):
    """
    Load training examples from ground truth annotations.
    
    The ground truth should be a JSON file with structure:
    {
        "positive_examples": [
            "Assignment 1: Read Chapter 3...",
            "Midterm Exam - October 15...",
            ...
        ],
        "negative_examples": [
            "Office Hours: Monday 2-4pm...",
            "Grading Policy: Exams 40%...",
            ...
        ]
    }
    
    Args:
        ground_truth_path: Path to ground truth JSON file
        
    Returns:
        Tuple of (positive_examples, negative_examples)
    """
    try:
        with open(ground_truth_path, 'r') as f:
            data = json.load(f)
        
        positive = data.get('positive_examples', [])
        negative = data.get('negative_examples', [])
        
        print(f"Loaded {len(positive)} positive examples")
        print(f"Loaded {len(negative)} negative examples")
        
        return positive, negative
        
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        print("\nTo train the classifier, you need to create a ground truth file.")
        print("See README.md for the expected format.")
        return [], []


def train_and_save_model(ground_truth_path, model_output_path='classifier.pkl'):
    """
    Train the Random Forest classifier and save it.
    
    Args:
        ground_truth_path: Path to training data
        model_output_path: Where to save trained model
    """
    # Load training data
    positive, negative = load_training_data(ground_truth_path)
    
    if not positive or not negative:
        print("\nCannot train without data. Exiting.")
        return
    
    # Initialize parser
    parser = SyllabusParser()
    
    # Train classifier
    print("\nTraining Random Forest classifier...")
    parser.train_classifier(positive, negative)
    
    # Save trained model
    with open(model_output_path, 'wb') as f:
        pickle.dump({
            'vectorizer': parser.vectorizer,
            'classifier': parser.classifier
        }, f)
    
    print(f"\nTrained model saved to {model_output_path}")
    print("\nModel ready for evaluation.")


def example_ground_truth_format():
    """
    Print example ground truth format.
    """
    print("Example ground truth format (ground_truth.json):")
    print("-" * 60)
    
    example = {
        "positive_examples": [
            "Assignment 1: Read Chapter 3 and submit 2-page response",
            "Midterm Exam - October 15, covers weeks 1-7",
            "Final Project Proposal due November 1",
            "Problem Set 3: Complete exercises 1-10"
        ],
        "negative_examples": [
            "Office Hours: Tuesdays 2-4pm in Room 301",
            "Grading: Exams 40%, Papers 30%, Participation 30%",
            "Required Textbook: Introduction to Algorithms",
            "Course Description: This course covers..."
        ]
    }
    
    print(json.dumps(example, indent=2))
    print("\nCreate a file like this with your annotated syllabi data.")


if __name__ == '__main__':
    import sys
    
    print("Assignment Classifier Training")
    print("=" * 60)
    print()
    
    if len(sys.argv) < 2:
        print("Usage: python train_classifier.py <ground_truth.json>")
        print()
        print("The ground truth file should contain annotated assignment")
        print("examples extracted from your syllabi corpus.")
        print()
        example_ground_truth_format()
    else:
        ground_truth_path = sys.argv[1]
        train_and_save_model(ground_truth_path)
