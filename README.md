# ToDue: A Syllabus Parsing & Deadline Automation Tool

A hybrid information extraction system designed to automatically parse course syllabi
and extract assignment deadlines to minimize manual and sometimes tedious process of logging assignments from all syllabi for the semester. 

# Requirements

Python 3.8+
pdfplumber
scikit-learn
dateparser
pandas
numpy

# Installation
```python
bashpip install pdfplumber scikit-learn dateparser pandas numpy
```
# Usage
```python
pythonfrom syllabus_parser import SyllabusParser

parser = SyllabusParser()
assignments = parser.parse_syllabus('path/to/syllabus.pdf')

for assignment in assignments:
    print(f"{assignment['date']}: {assignment['name']}")
```

# Limitations
- Performance degrades on syllabi with complex table structures
- Cannot resolve relative temporal expressions ("three weeks after midterm")
- Limited to assignment name and due date extraction (does not extract submission methods, point values, or detailed requirements)

