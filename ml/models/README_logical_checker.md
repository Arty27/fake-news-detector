# Logical Inconsistency Checker

## Overview
The Logical Inconsistency Checker is a separate module that identifies logically impossible entity combinations in text. It works alongside the existing BERT-based fake news detection system to catch obvious logical impossibilities that the ML model might miss.

## What It Detects

### 1. **Fictional Character Participation**
- Fictional characters (Harry Potter, Sherlock Holmes, etc.) participating in real-world events
- Example: "Harry Potter won the cricket worldcup" → **INCONSISTENT**

### 2. **Sport Mismatches**
- Incompatible sports combinations in the same context
- Example: "Won cricket worldcup during FIFA Championship" → **INCONSISTENT**

### 3. **Temporal Impossibilities**
- Conflicting time periods mentioned together
- Example: "Ancient Romans used smartphones in 2024" → **INCONSISTENT**

## How to Use

### Basic Usage
```python
from models.logical_inconsistency_checker import LogicalInconsistencyChecker

checker = LogicalInconsistencyChecker()
result = checker.check("Your text here")

print(f"Inconsistency Score: {result['overall_inconsistency_score']}")
print(f"Is Consistent: {result['is_logically_consistent']}")
```

### Integration with Main System
The checker is already integrated in `main.py` and will run automatically when you test articles.

## Output Format

```python
{
    "inconsistencies": [LogicalInconsistency objects],
    "inconsistency_count": 2,
    "overall_inconsistency_score": 0.85,
    "is_logically_consistent": False,
    "confidence": 0.92
}
```

## How to Remove (If Needed)

### Option 1: Remove from main.py
```python
# Remove this line:
from models.logical_inconsistency_checker import LogicalInconsistencyChecker

# Remove this line:
logical_checker = LogicalInconsistencyChecker()

# Remove the entire logical checking section
```

### Option 2: Comment Out
```python
# from models.logical_inconsistency_checker import LogicalInconsistencyChecker
# logical_checker = LogicalInconsistencyChecker()
```

### Option 3: Delete Files
- Delete `ml/models/logical_inconsistency_checker.py`
- Delete `ml/test_logical_checker.py`
- Delete `ml/models/README_logical_checker.md`

## Customization

### Add New Fictional Characters
```python
# In _init_fictional_characters method
self.fictional_characters = {
    "harry potter": "fictional_wizard",
    "your_character": "fictional_type",
    # ... add more
}
```

### Add New Sports Categories
```python
# In _init_sports_knowledge method
self.sports_categories = {
    "cricket": ["cricket", "worldcup", "ipl"],
    "your_sport": ["term1", "term2"],
    # ... add more
}
```

### Modify Confidence Thresholds
```python
# In _calculate_inconsistency_score method
if overall_score < 0.3:  # Change this threshold
    return "is_logically_consistent": True
```

## Testing

Run the test script to see examples:
```bash
cd ml
python test_logical_checker.py
```

## Benefits

1. **Catches Obvious Fakes**: Identifies logically impossible statements
2. **Explainable**: Shows exactly why something is inconsistent
3. **Modular**: Easy to add/remove without affecting other components
4. **Fast**: Rule-based approach, no ML inference needed
5. **Customizable**: Easy to add new rules and knowledge bases

## Limitations

1. **Rule-Based**: Only catches predefined logical impossibilities
2. **Domain-Specific**: Currently focused on sports and fictional characters
3. **Language Dependent**: Works best with English text
4. **Static Knowledge**: Requires manual updates for new domains

## Future Enhancements

1. **Machine Learning Integration**: Train models to detect new types of inconsistencies
2. **Knowledge Graph**: Use external knowledge bases for more comprehensive checking
3. **Multi-Language Support**: Extend to other languages
4. **Real-time Updates**: Connect to live knowledge sources

