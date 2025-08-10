#!/usr/bin/env python3
"""Enhanced CLI interface with better user experience."""

import sys
import os
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def process_email_enhanced(content: str) -> dict:
    """Process email with enhanced error handling and validation."""
    try:
        from crewai_email_triage.core import process_email
        from crewai_email_triage.basic_validation import validate_email_basic
        
        # Validate input
        validation_result = validate_email_basic(content)
        
        # Process email
        processed_result = process_email(content)
        
        return {
            "processed_content": processed_result,
            "validation": validation_result,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "processed_content": None,
            "validation": None,
            "success": False,
            "error": str(e)
        }

def main():
    """Enhanced CLI main function."""
    parser = argparse.ArgumentParser(description="Enhanced CrewAI Email Triage")
    parser.add_argument("--message", help="Email content to process")
    parser.add_argument("--file", help="File containing email content")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, don't process")
    
    args = parser.parse_args()
    
    # Get content
    if args.file:
        with open(args.file) as f:
            content = f.read()
    elif args.message:
        content = args.message
    else:
        content = input("Enter email content: ")
    
    # Process
    if args.validate_only:
        from crewai_email_triage.basic_validation import validate_email_basic
        result = validate_email_basic(content)
    else:
        result = process_email_enhanced(content)
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if args.validate_only:
            print(f"Valid: {result['is_valid']}")
            if result['warnings']:
                print("Warnings:")
                for warning in result['warnings']:
                    print(f"  - {warning}")
        else:
            if result['success']:
                print(f"Result: {result['processed_content']}")
                if result['validation']['warnings']:
                    print("Warnings:")
                    for warning in result['validation']['warnings']:
                        print(f"  - {warning}")
            else:
                print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()
