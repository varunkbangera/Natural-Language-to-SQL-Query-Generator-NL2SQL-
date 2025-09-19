# src/nlp_engine/schema_mapper.py

from fuzzywuzzy import fuzz
from typing import Dict, List, Optional, Any, Tuple
import re

class SchemaMapper:
    """
    Maps natural language terms to actual database schema elements
    """
    
    def __init__(self, schema_info: Dict[str, Any]):
        self.schema_info = schema_info
        self.columns = schema_info.get('columns', {})
        self.column_names = list(self.columns.keys())
        self.numeric_columns = schema_info.get('numeric_columns', [])
        self.text_columns = schema_info.get('text_columns', [])
        self.date_columns = schema_info.get('date_columns', [])
        
        # Build synonym mapping
        self.column_synonyms = self._build_column_synonyms()
    
    def _build_column_synonyms(self) -> Dict[str, List[str]]:
        """Build synonym mapping for column names"""
        synonyms = {}
        
        for col in self.column_names:
            col_lower = col.lower()
            synonyms[col] = [col_lower]
            
            # Common synonyms based on column names
            if 'name' in col_lower:
                synonyms[col].extend(['names', 'title', 'label', 'employee', 'person'])
            elif 'id' in col_lower:
                synonyms[col].extend(['identifier', 'number', 'code'])
            elif 'age' in col_lower:
                synonyms[col].extend(['years', 'old'])
            elif 'salary' in col_lower:
                synonyms[col].extend(['pay', 'wage', 'income', 'earnings', 'money'])
            elif 'price' in col_lower:
                synonyms[col].extend(['cost', 'amount', 'value', 'money'])
            elif 'date' in col_lower:
                synonyms[col].extend(['time', 'when'])
            elif 'department' in col_lower:
                synonyms[col].extend(['dept', 'division', 'team'])
            elif 'revenue' in col_lower:
                synonyms[col].extend(['sales', 'income', 'earnings'])
            elif 'score' in col_lower:
                synonyms[col].extend(['grade', 'rating', 'points'])
            elif 'customer' in col_lower:
                synonyms[col].extend(['client', 'buyer'])
            elif 'product' in col_lower:
                synonyms[col].extend(['item', 'goods'])
        
        return synonyms
    
    def map_column_references(self, query_tokens: List[str]) -> List[str]:
        """Map tokens to actual column names"""
        mapped_columns = []
        
        for token in query_tokens:
            column_match = self._find_column_match(token)
            if column_match:
                mapped_columns.append(column_match)
        
        return mapped_columns
    
    def _find_column_match(self, token: str) -> Optional[str]:
        """Find best matching column for a token"""
        token_lower = token.lower()
        
        # Direct exact match
        for col in self.column_names:
            if token_lower == col.lower():
                return col
        
        # Synonym match
        for col, synonyms in self.column_synonyms.items():
            if token_lower in synonyms:
                return col
        
        # Fuzzy match (70% similarity)
        best_score = 0
        best_match = None
        
        for col in self.column_names:
            score = fuzz.ratio(token_lower, col.lower())
            if score > best_score and score > 70:
                best_score = score
                best_match = col
        
        return best_match
    
    def extract_conditions(self, processed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract WHERE conditions from processed query"""
        conditions = []
        
        # Get comparison operators from processed query
        comparisons = processed_query.get('comparison_operators', [])
        numbers = processed_query.get('numbers', [])
        keywords = processed_query.get('keywords', [])
        
        for comp in comparisons:
            # Find column for this comparison
            column = self._find_condition_column(keywords, comp)
            if column:
                conditions.append({
                    'column': column,
                    'operator': comp['sql_operator'],
                    'value': comp['value'],
                    'type': self._get_column_type(column)
                })
        
        return conditions
    
    def _find_condition_column(self, keywords: List[str], comparison: Dict) -> Optional[str]:
        """Find which column a comparison applies to"""
        # Look for column references in keywords
        for keyword in keywords:
            column_match = self._find_column_match(keyword)
            if column_match:
                # Check if this column can be used with the comparison value
                if self._is_valid_comparison(column_match, comparison['value']):
                    return column_match
        
        # Default: use first numeric column for numeric comparisons
        if self.numeric_columns and isinstance(comparison['value'], (int, float)):
            return self.numeric_columns[0]
        
        return None
    
    def _is_valid_comparison(self, column: str, value: Any) -> bool:
        """Check if a comparison makes sense for this column type"""
        column_type = self._get_column_type(column)
        
        if column_type in ['INTEGER', 'REAL'] and isinstance(value, (int, float)):
            return True
        elif column_type == 'TEXT' and isinstance(value, str):
            return True
        
        return False
    
    def _get_column_type(self, column: str) -> str:
        """Get the data type of a column"""
        if column in self.columns:
            return self.columns[column].get('type', 'TEXT')
        return 'TEXT'
    
    def suggest_aggregation_column(self, intent: str) -> Optional[str]:
        """Suggest which column to use for aggregation functions"""
        if intent in ['SUM', 'AVG', 'MAX', 'MIN']:
            # Prefer numeric columns for math operations
            if self.numeric_columns:
                # Look for salary, price, revenue, score type columns first
                for col in self.numeric_columns:
                    col_lower = col.lower()
                    if any(word in col_lower for word in ['salary', 'price', 'revenue', 'amount', 'score', 'cost']):
                        return col
                # Otherwise return first numeric column
                return self.numeric_columns[0]
        
        elif intent == 'COUNT':
            # For COUNT, we can use any column or just COUNT(*)
            return None  # Will default to COUNT(*)
        
        return None
    
    def get_suggested_queries(self) -> List[str]:
        """Generate suggested queries based on schema"""
        suggestions = [
            "Show all records",
            "Count total rows"
        ]
        
        # Add column-specific suggestions
        for col in self.column_names[:3]:  # First 3 columns
            suggestions.append(f"Show all {col} values")
        
        # Add numeric column suggestions
        for col in self.numeric_columns[:2]:  # First 2 numeric columns
            suggestions.extend([
                f"What is the average {col}?",
                f"Show records with {col} > 100"
            ])
        
        return suggestions
    
    def validate_query_feasibility(self, processed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the processed query can be executed with current schema"""
        issues = []
        suggestions = []
        
        intent = processed_query.get('intent', 'SELECT')
        keywords = processed_query.get('keywords', [])
        comparisons = processed_query.get('comparison_operators', [])
        
        # Check if aggregation makes sense
        if intent in ['SUM', 'AVG', 'MAX', 'MIN'] and not self.numeric_columns:
            issues.append(f"Cannot calculate {intent} - no numeric columns found")
            suggestions.append("Try counting records instead")
        
        # Check if column references make sense
        mapped_columns = self.map_column_references(keywords)
        if keywords and not mapped_columns:
            issues.append("Could not find matching columns for your query")
            suggestions.append(f"Available columns: {', '.join(self.column_names)}")
        
        # Check if comparisons make sense
        conditions = self.extract_conditions(processed_query)
        if comparisons and not conditions:
            issues.append("Could not understand the filter conditions")
            suggestions.append("Try simpler conditions like 'salary > 50000'")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'suggestions': suggestions,
            'mapped_columns': mapped_columns,
            'conditions': conditions
        }


# Test the schema mapper
if __name__ == "__main__":
    # Test with sample schema
    sample_schema = {
        'columns': {
            'id': {'type': 'INTEGER'},
            'name': {'type': 'TEXT'},
            'age': {'type': 'INTEGER'},
            'salary': {'type': 'REAL'},
            'department': {'type': 'TEXT'}
        },
        'numeric_columns': ['id', 'age', 'salary'],
        'text_columns': ['name', 'department']
    }
    
    mapper = SchemaMapper(sample_schema)
    
    # Test column mapping
    print("Testing column mapping:")
    test_tokens = ['employees', 'salary', 'pay', 'department', 'age']
    for token in test_tokens:
        match = mapper._find_column_match(token)
        print(f"'{token}' -> '{match}'")
    
    # Test query suggestions
    print("\nSuggested queries:")
    suggestions = mapper.get_suggested_queries()
    for suggestion in suggestions:
        print(f"- {suggestion}")