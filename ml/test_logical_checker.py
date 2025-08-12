#!/usr/bin/env python3
"""
Simple test script for the Logical Inconsistency Checker
Run this to see how it detects logical impossibilities.
"""

from models.logical_inconsistency_checker import LogicalInconsistencyChecker

def test_logical_checker():
    """Test the logical inconsistency checker with various examples."""
    
    checker = LogicalInconsistencyChecker()
    
    # Test cases
    test_cases = [
        "Vijay Kumar won the cricket worldcup with Harry Potter during the FIFA Championship!",
        "Einstein won the 2024 Olympics in quantum physics",
        "Harry Potter became US President in 2020",
        "Dinosaurs built the pyramids in 2023",
        "Manchester United won the cricket championship",
        "The sun rises in the west and sets in the east",
        "A normal news article about weather in London",
        "Sherlock Holmes solved the mystery of the 2024 election",
        "Basketball players competed in the Wimbledon tennis tournament"
    ]
    
    print("=== Logical Inconsistency Checker Test ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case}")
        print("-" * 60)
        
        try:
            result = checker.check(test_case)
            
            print(f"Inconsistency Count: {result['inconsistency_count']}")
            print(f"Overall Score: {result['overall_inconsistency_score']:.3f}")
            print(f"Logically Consistent: {result['is_logically_consistent']}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            if result['inconsistencies']:
                print("\nDetected Inconsistencies:")
                for j, inc in enumerate(result['inconsistencies'], 1):
                    print(f"  {j}. {inc.description}")
                    print(f"     Type: {inc.inconsistency_type}")
                    print(f"     Entities: {', '.join(inc.entities_involved)}")
                    print(f"     Rule: {inc.rule_applied}")
            else:
                print("\nNo logical inconsistencies detected.")
                
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    test_logical_checker()

