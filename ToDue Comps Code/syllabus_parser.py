#!/usr/bin/env python3
"""
ToDue: Syllabus Assignment Extraction System
Automated extraction of assignment deadlines from course syllabi using hybrid
rule-based and machine learning approaches.
"""

import re
import pdfplumber
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class SyllabusParser:
    """
    Extract assignments and due dates from course syllabi using a hybrid approach
    combining layout-aware PDF parsing, temporal expression recognition, and
    Random Forest classification.
    """
    
    # Assignment-related keywords for rule-based detection
    ASSIGNMENT_KEYWORDS = [
        'assignment', 'homework', 'hw', 'paper', 'essay', 'project', 'exam',
        'midterm', 'final', 'quiz', 'test', 'presentation', 'discussion',
        'reading', 'response', 'reflection', 'due', 'submit', 'deadline',
        'deliverable', 'lab', 'report', 'problem set', 'pset'
    ]
    
    # Temporal expression patterns (regex for date detection)
    DATE_PATTERNS = [
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b',
        r'\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b',
        r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
    ]
    
    def __init__(self, default_year=2025):
        """
        Initialize parser with default year for date normalization.
        
        Args:
            default_year: Year to use when dates don't specify year
        """
        self.default_year = default_year
        self.vectorizer = None
        self.classifier = None
        self.trained = False
        
    def extract_text_blocks(self, pdf_path):
        """
        Extract text from PDF with layout-aware parsing using pdfplumber.
        Preserves structural information about text positioning and formatting.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of text blocks with metadata (page, position, text)
        """
        text_blocks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract words with layout information
                words = page.extract_words()
                
                # Group words into lines based on vertical position
                lines = {}
                for word in words:
                    y_pos = round(word['top'])
                    if y_pos not in lines:
                        lines[y_pos] = []
                    lines[y_pos].append(word)
                
                # Sort lines by position and extract text
                for y_pos in sorted(lines.keys()):
                    line_words = sorted(lines[y_pos], key=lambda w: w['x0'])
                    text = ' '.join(w['text'] for w in line_words)
                    
                    if text.strip():
                        # Extract layout features
                        avg_height = np.mean([w['height'] for w in line_words])
                        x_pos = line_words[0]['x0']
                        
                        text_blocks.append({
                            'page': page_num + 1,
                            'y_position': y_pos,
                            'x_position': x_pos,
                            'font_height': avg_height,
                            'text': text.strip()
                        })
        
        return text_blocks
    
    def detect_dates(self, text):
        """
        Extract temporal expressions from text using regex patterns.
        
        Args:
            text: Input text string
            
        Returns:
            List of detected dates with positions
        """
        dates = []
        
        for pattern in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                date_str = match.group()
                normalized = self._normalize_date(date_str)
                
                if normalized:
                    dates.append({
                        'raw': date_str,
                        'normalized': normalized,
                        'position': match.start()
                    })
        
        return dates
    
    def _normalize_date(self, date_str):
        """
        Normalize date string to YYYY-MM-DD format.
        
        Args:
            date_str: Raw date string
            
        Returns:
            Normalized date string or None if parsing fails
        """
        try:
            # Handle common date formats
            for fmt in ['%B %d', '%b %d', '%m/%d/%Y', '%m/%d', '%m-%d-%Y', '%m-%d']:
                try:
                    parsed = datetime.strptime(date_str.strip(), fmt)
                    # Add default year if not present
                    if parsed.year == 1900:
                        parsed = parsed.replace(year=self.default_year)
                    return parsed.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            return None
        except:
            return None
    
    def extract_layout_features(self, block):
        """
        Extract layout-aware features from text block for classification.
        
        Args:
            block: Text block dictionary with layout metadata
            
        Returns:
            Feature dictionary
        """
        text = block['text']
        text_lower = text.lower()
        
        features = {
            # Text features
            'length': len(text),
            'word_count': len(text.split()),
            
            # Layout features
            'font_height': block.get('font_height', 0),
            'x_position': block.get('x_position', 0),
            'indentation': block.get('x_position', 0) > 50,
            
            # Format features (tables, bullets, prose)
            'has_bullet': any(c in text[:3] for c in ['•', '○', '-', '*']),
            'has_number_prefix': bool(re.match(r'^\d+\.', text)),
            'has_pipe': '|' in text,  # Table indicator
            
            # Assignment keywords
            'keyword_count': sum(1 for kw in self.ASSIGNMENT_KEYWORDS if kw in text_lower),
            
            # Punctuation
            'has_colon': ':' in text,
            'has_dash': '-' in text or '–' in text,
        }
        
        return features
    
    def train_classifier(self, positive_examples, negative_examples):
        """
        Train Random Forest classifier on labeled examples.
        
        Args:
            positive_examples: List of assignment text strings
            negative_examples: List of non-assignment text strings
        """
        # Prepare training data
        texts = positive_examples + negative_examples
        labels = [1] * len(positive_examples) + [0] * len(negative_examples)
        
        # TF-IDF vectorization for text features
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words='english'
        )
        
        X_text = self.vectorizer.fit_transform(texts)
        
        # Train Random Forest
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        
        self.classifier.fit(X_text, labels)
        self.trained = True
        
        # Report training accuracy
        accuracy = self.classifier.score(X_text, labels)
        print(f"Classifier trained. Training accuracy: {accuracy:.1%}")
    
    def segment_by_dates(self, text_blocks):
        """
        Segment text blocks by detected dates to create assignment candidates.
        
        Args:
            text_blocks: List of text blocks with layout info
            
        Returns:
            List of segments (text + associated date)
        """
        segments = []
        current_segment = []
        current_date = None
        
        for block in text_blocks:
            dates = self.detect_dates(block['text'])
            
            if dates:
                # Save previous segment
                if current_segment and current_date:
                    segments.append({
                        'text': ' '.join(b['text'] for b in current_segment),
                        'blocks': current_segment,
                        'date': current_date
                    })
                
                # Start new segment
                current_segment = [block]
                current_date = dates[0]['normalized']
            else:
                # Continue building segment
                current_segment.append(block)
        
        # Don't forget last segment
        if current_segment and current_date:
            segments.append({
                'text': ' '.join(b['text'] for b in current_segment),
                'blocks': current_segment,
                'date': current_date
            })
        
        return segments
    
    def classify_segments(self, segments):
        """
        Classify segments as assignments using trained Random Forest.
        
        Args:
            segments: List of text segments
            
        Returns:
            List of segments classified as assignments
        """
        if not self.trained:
            print("Warning: Classifier not trained. Using rule-based fallback.")
            return self._rule_based_classification(segments)
        
        assignments = []
        
        for segment in segments:
            # Vectorize text
            X = self.vectorizer.transform([segment['text']])
            
            # Predict
            prediction = self.classifier.predict(X)[0]
            confidence = self.classifier.predict_proba(X)[0][1]
            
            if prediction == 1 and confidence > 0.5:
                assignments.append({
                    'text': segment['text'][:500],  # Truncate for display
                    'date': segment['date'],
                    'confidence': confidence
                })
        
        return assignments
    
    def _rule_based_classification(self, segments):
        """
        Fallback rule-based classification when ML not trained.
        
        Args:
            segments: List of text segments
            
        Returns:
            List of segments classified as assignments
        """
        assignments = []
        
        for segment in segments:
            text_lower = segment['text'].lower()
            
            # Check for assignment indicators
            has_keyword = any(kw in text_lower for kw in self.ASSIGNMENT_KEYWORDS)
            has_due_pattern = bool(re.search(r'due|submit|deadline', text_lower))
            
            if has_keyword or has_due_pattern:
                assignments.append({
                    'text': segment['text'][:500],
                    'date': segment['date'],
                    'confidence': 0.7  # Default confidence
                })
        
        return assignments
    
    def parse_syllabus(self, pdf_path, course_name=None):
        """
        Main parsing function to extract assignments from syllabus.
        
        Args:
            pdf_path: Path to syllabus PDF
            course_name: Optional course identifier
            
        Returns:
            List of extracted assignments with dates
        """
        # Step 1: Layout-aware PDF extraction
        text_blocks = self.extract_text_blocks(pdf_path)
        
        # Step 2: Segment by dates
        segments = self.segment_by_dates(text_blocks)
        
        # Step 3: Classify assignments
        assignments = self.classify_segments(segments)
        
        # Add course name
        for assignment in assignments:
            assignment['course'] = course_name or 'Unknown'
        
        return assignments


def demo():
    """
    Demonstration of parser usage.
    """
    # Initialize parser
    parser = SyllabusParser(default_year=2025)
    
    # Example training data (in practice, load from ground truth)
    positive_examples = [
        "Assignment 1: Read Chapter 3 and submit response",
        "Midterm Exam - covers weeks 1-7",
        "Final Project Proposal due",
        "Homework 3: Problem Set on Linear Algebra"
    ]
    
    negative_examples = [
        "Office Hours: Monday 2-4pm",
        "Course Description: This course covers...",
        "Grading Policy: Exams 40%, Papers 30%, Participation 30%",
        "Textbook: Introduction to Computer Science"
    ]
    
    # Train classifier
    parser.train_classifier(positive_examples, negative_examples)
    
    # Parse syllabus (example - replace with actual file path)
    # assignments = parser.parse_syllabus('path/to/syllabus.pdf', 'CS 101')
    
    print("\nParser initialized and trained.")
    print("Usage: assignments = parser.parse_syllabus('syllabus.pdf', 'Course Name')")


if __name__ == '__main__':
    demo()
