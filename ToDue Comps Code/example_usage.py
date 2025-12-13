#!/usr/bin/env python3
"""
Example usage of the ToDue syllabus parser.

Demonstrates how to use the parser on your own syllabi.
Note: Requires training data (annotated syllabi) to achieve reported performance.
"""

from syllabus_parser import SyllabusParser


def basic_usage():
    """Basic example: parse a single syllabus."""
    
    # Initialize parser
    parser = SyllabusParser(default_year=2025)
    
    # Option 1: Use without training (rule-based fallback)
    print("Example 1: Rule-based parsing (no training)")
    print("-" * 60)
    
    # Parse syllabus
    assignments = parser.parse_syllabus(
        pdf_path='path/to/your/syllabus.pdf',
        course_name='Introduction to Computer Science'
    )
    
    # Display results
    for assignment in assignments:
        print(f"Date: {assignment['date']}")
        print(f"Text: {assignment['text'][:100]}...")
        print(f"Confidence: {assignment['confidence']:.2f}")
        print()


def trained_usage():
    """Example with training data for better performance."""
    
    print("\nExample 2: With classifier training")
    print("-" * 60)
    
    # Initialize parser
    parser = SyllabusParser(default_year=2025)
    
    # Prepare training examples
    # (In practice, these would come from your annotated ground truth)
    positive_examples = [
        "Assignment 1: Read Chapter 3 and submit a 2-page response",
        "Midterm Exam - Covers material from weeks 1-7",
        "Final Project Proposal due by end of week",
        "Problem Set 3: Complete exercises 1-10 from textbook",
        "Research Paper: 10-15 pages on approved topic"
    ]
    
    negative_examples = [
        "Office Hours: Tuesdays and Thursdays 2-4pm in Room 301",
        "Course Description: This course provides an introduction to...",
        "Grading Breakdown: Exams 40%, Papers 30%, Participation 30%",
        "Required Textbook: Introduction to Algorithms, 3rd Edition",
        "Attendance Policy: Students are expected to attend all lectures"
    ]
    
    # Train classifier
    parser.train_classifier(positive_examples, negative_examples)
    
    # Now parse with trained classifier
    assignments = parser.parse_syllabus(
        pdf_path='path/to/your/syllabus.pdf',
        course_name='Data Structures'
    )
    
    print(f"Found {len(assignments)} assignments")


def batch_processing():
    """Example: process multiple syllabi at once."""
    
    print("\nExample 3: Batch processing")
    print("-" * 60)
    
    # Initialize and train parser
    parser = SyllabusParser(default_year=2025)
    
    # Training data (abbreviated for example)
    positive = ["Assignment due next week", "Midterm exam scheduled"]
    negative = ["Office hours available", "Textbook required"]
    parser.train_classifier(positive, negative)
    
    # List of syllabi to process
    syllabi = [
        {'path': 'syllabus_cs101.pdf', 'name': 'CS 101'},
        {'path': 'syllabus_math201.pdf', 'name': 'Math 201'},
        {'path': 'syllabus_phys150.pdf', 'name': 'Physics 150'},
    ]
    
    # Process each syllabus
    all_assignments = []
    for syllabus in syllabi:
        try:
            assignments = parser.parse_syllabus(
                syllabus['path'],
                syllabus['name']
            )
            all_assignments.extend(assignments)
            print(f"{syllabus['name']}: {len(assignments)} assignments found")
        except Exception as e:
            print(f"{syllabus['name']}: Error - {e}")
    
    # Sort all assignments by date
    all_assignments.sort(key=lambda x: x['date'])
    
    print(f"\nTotal: {len(all_assignments)} assignments across all courses")


def export_to_calendar():
    """Example: export assignments to calendar format."""
    
    print("\nExample 4: Export to calendar")
    print("-" * 60)
    
    parser = SyllabusParser(default_year=2025)
    
    # Parse syllabus
    assignments = parser.parse_syllabus('syllabus.pdf', 'Biology 101')
    
    # Create calendar events (pseudo-code)
    print("Calendar entries:")
    for assignment in assignments:
        print(f"Event: {assignment['text'][:50]}")
        print(f"Date: {assignment['date']}")
        print(f"Course: {assignment['course']}")
        print()


if __name__ == '__main__':
    print("ToDue Syllabus Parser - Usage Examples")
    print("=" * 60)
    print()
    print("Note: These examples use placeholder file paths.")
    print("Replace 'path/to/your/syllabus.pdf' with actual PDF files.")
    print()
    
    # Run examples (commented out since we don't have actual PDFs)
    # basic_usage()
    # trained_usage()
    # batch_processing()
    # export_to_calendar()
    
    print("Uncomment the function calls above to run examples with your data.")
