#!/usr/bin/env python3
"""
Test script for the improved query builder
"""

from models.query_builder import QueryBuilder

def test_improved_query_builder():
    """Test the improved query builder with the problematic sentence."""
    
    qb = QueryBuilder()
    
    # Test the problematic sentence
    test_text = "The incredible hulk is one of the best left handed batsman in rugby"
    
    print("=== Testing Improved Query Builder ===")
    print(f"Input text: {test_text}")
    print("-" * 60)
    
    try:
        queries = qb.build(test_text)
        
        print(f"Generated queries ({len(queries)}):")
        for i, query in enumerate(queries, 1):
            print(f"{i}. '{query}' (length: {len(query)})")
        
        print(f"\nQuery quality analysis:")
        for i, query in enumerate(queries, 1):
            if len(query) <= 2:
                print(f"  {i}. ❌ Too short: '{query}'")
            elif query.lower() in ['one', 'the', 'a', 'an', 'and', 'or', 'but']:
                print(f"  {i}. ❌ Meaningless: '{query}'")
            elif len(query) > 50:
                print(f"  {i}. ⚠️  Too long: '{query}'")
            else:
                print(f"  {i}. ✅ Good: '{query}'")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_improved_query_builder()
