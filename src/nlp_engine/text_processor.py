# src/nlp_engine/text_processor.py

import re
from typing import Dict, List, Optional, Any

class TextProcessor:
    def __init__(self):
        # Simple intent patterns
        self.intent_patterns = {
            'SELECT': ['show', 'display', 'list', 'get', 'find'],
            'COUNT': ['how many', 'count', 'number of'],
            'SUM': ['total', 'sum'],
            'AVG': ['average', 'mean'],
            'MAX': ['maximum', 'max', 'highest'],
            'MIN': ['minimum', 'min', 'lowest']
        }
        
        self.comparison_patterns = {
            'greater_than': ['greater than', 'more than', 'above', 'over', '>'],
            'less_than': ['less than', 'below', 'under', '<'],
            'equal': ['equal to', 'equals', '=', 'is']
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        if not query.strip():
            return self._empty_result()
        
        # Clean the query
        cleaned_query = query.lower().strip()
        
        # Extract components
        result = {
            'original_query': query,
            'cleaned_query': cleaned_query,
            'tokens': cleaned_query.split(),
            'intent': self._extract_intent(cleaned_query),
            'numbers': self._extract_numbers(cleaned_query),
            'comparison_operators': self._extract_comparisons(cleaned_query),
            'keywords': self._extract_keywords(cleaned_query),
            'potential_columns': self._extract_potential_columns(cleaned_query)
        }
        
        return result
    
    def _extract_intent(self, query: str) -> str:
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    return intent
        return 'SELECT'
    
    def _extract_numbers(self, query: str) -> List[Dict[str, Any]]:
        numbers = []
        number_matches = re.finditer(r'\b\d+(?:\.\d+)?\b', query)
        
        for match in number_matches:
            numbers.append({
                'value': float(match.group()) if '.' in match.group() else int(match.group()),
                'text': match.group()
            })
        
        return numbers
    
    def _extract_comparisons(self, query: str) -> List[Dict[str, Any]]:
        comparisons = []
        
        for op_type, patterns in self.comparison_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    # Find number after the operator
                    remaining = query[query.find(pattern) + len(pattern):].strip()
                    number_match = re.search(r'\b\d+(?:\.\d+)?\b', remaining)
                    
                    if number_match:
                        value = float(number_match.group()) if '.' in number_match.group() else int(number_match.group())
                        comparisons.append({
                            'operator': op_type,
                            'value': value,
                            'sql_operator': '>' if op_type == 'greater_than' else '<' if op_type == 'less_than' else '='
                        })
        
        return comparisons
    
    def _extract_keywords(self, query: str) -> List[str]:
        # Simple keyword extraction
        words = query.split()
        keywords = []
        
        stop_words = ['the', 'is', 'are', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'all', 'me']
        
        for word in words:
            if word not in stop_words and len(word) > 2 and word.isalpha():
                keywords.append(word)
        
        return keywords
    
    def _extract_potential_columns(self, query: str) -> List[str]:
        # Common column names
        column_words = ['name', 'id', 'age', 'salary', 'price', 'date', 'email', 'phone', 'city', 'department', 'title', 'status']
        
        potential_columns = []
        words = query.split()
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in column_words:
                potential_columns.append(clean_word)
        
        return potential_columns
    
    def _empty_result(self) -> Dict[str, Any]:
        return {
            'original_query': '',
            'cleaned_query': '',
            'tokens': [],
            'intent': 'SELECT',
            'numbers': [],
            'comparison_operators': [],
            'keywords': [],
            'potential_columns': []
        }
    
    def get_query_suggestions(self, schema_info: Dict[str, Any]) -> List[str]:
        suggestions = ["Show all records", "Count total rows"]
        
        if schema_info and 'columns' in schema_info:
            columns = list(schema_info['columns'].keys())
            if columns:
                suggestions.append(f"Show all {columns[0]} values")
            
            numeric_cols = schema_info.get('numeric_columns', [])
            if numeric_cols:
                suggestions.append(f"What is the average {numeric_cols[0]}?")
        
        return suggestions