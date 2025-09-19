# src/nlp_engine/confidence_scorer.py

from typing import Dict, List, Any, Tuple
import re

class QueryConfidenceScorer:
    """
    Advanced confidence scoring for generated queries
    """
    
    def __init__(self, schema_info: Dict[str, Any]):
        self.schema_info = schema_info
    
    def calculate_comprehensive_confidence(
        self, 
        question: str,
        intent: str,
        mapped_columns: List[str],
        conditions: List[Dict[str, Any]],
        generated_sql: str
    ) -> Dict[str, Any]:
        """Calculate comprehensive confidence score with detailed breakdown"""
        
        scores = {
            'intent_confidence': self._score_intent_recognition(question, intent),
            'column_mapping_confidence': self._score_column_mapping(question, mapped_columns),
            'condition_extraction_confidence': self._score_condition_extraction(question, conditions),
            'sql_validity_confidence': self._score_sql_validity(generated_sql),
            'semantic_coherence_confidence': self._score_semantic_coherence(question, intent, mapped_columns)
        }
        
        # Calculate weighted overall confidence
        weights = {
            'intent_confidence': 0.25,
            'column_mapping_confidence': 0.30,
            'condition_extraction_confidence': 0.20,
            'sql_validity_confidence': 0.15,
            'semantic_coherence_confidence': 0.10
        }
        
        overall_confidence = sum(scores[key] * weights[key] for key in scores.keys())
        
        # Generate detailed feedback
        feedback = self._generate_confidence_feedback(scores, overall_confidence)
        
        return {
            'overall_confidence': min(overall_confidence, 1.0),
            'detailed_scores': scores,
            'confidence_level': self._get_confidence_level(overall_confidence),
            'feedback': feedback,
            'improvement_suggestions': self._get_improvement_suggestions(scores, question)
        }
    
    def _score_intent_recognition(self, question: str, intent: str) -> float:
        """Score intent recognition accuracy"""
        question_lower = question.lower()
        
        # High confidence patterns for each intent
        high_confidence_patterns = {
            'COUNT': [r'\bhow many\b', r'\bcount\b', r'\bnumber of\b'],
            'AVG': [r'\baverage\b', r'\bmean\b'],
            'MAX': [r'\bmaximum\b', r'\bhighest\b', r'\bmax\b'],
            'MIN': [r'\bminimum\b', r'\blowest\b', r'\bmin\b'],
            'SUM': [r'\btotal\b', r'\bsum\b'],
            'SELECT': [r'\bshow\b', r'\bdisplay\b', r'\blist\b']
        }
        
        if intent in high_confidence_patterns:
            for pattern in high_confidence_patterns[intent]:
                if re.search(pattern, question_lower):
                    return 0.95
        
        # Medium confidence - less specific patterns
        if intent == 'COUNT' and 'by' in question_lower:
            return 0.85
        if intent in ['AVG', 'MAX', 'MIN'] and any(word in question_lower for word in ['what', 'is']):
            return 0.80
        
        return 0.60  # Default confidence
    
    def _score_column_mapping(self, question: str, mapped_columns: List[str]) -> float:
        """Score column mapping accuracy"""
        if not mapped_columns:
            return 0.30  # Low confidence if no columns mapped
        
        question_lower = question.lower()
        score = 0.0
        
        # Check for exact column name matches
        exact_matches = 0
        partial_matches = 0
        
        for col in mapped_columns:
            if col.lower() in question_lower:
                exact_matches += 1
            elif any(word in col.lower() for word in question_lower.split()):
                partial_matches += 1
        
        # Calculate score based on matches
        if exact_matches > 0:
            score = 0.90 + (exact_matches - 1) * 0.05  # Bonus for multiple exact matches
        elif partial_matches > 0:
            score = 0.70 + (partial_matches - 1) * 0.03
        else:
            score = 0.50  # Moderate confidence for semantic matching
        
        return min(score, 1.0)
    
    def _score_condition_extraction(self, question: str, conditions: List[Dict[str, Any]]) -> float:
        """Score condition extraction accuracy"""
        question_lower = question.lower()
        
        # Check if question has comparison operators
        has_operators = any(op in question_lower for op in ['>', '<', '=', 'greater', 'less', 'above', 'below'])
        has_numbers = bool(re.search(r'\d+', question))
        
        if not has_operators and not has_numbers:
            return 1.0 if not conditions else 0.7  # No conditions needed
        
        if has_operators and has_numbers and conditions:
            # Validate condition makes sense
            for condition in conditions:
                col_info = self.schema_info['columns'].get(condition['column'], {})
                if col_info.get('type') in ['INTEGER', 'REAL']:
                    return 0.90  # High confidence for numeric conditions
            return 0.75  # Medium confidence
        
        if has_operators or has_numbers:
            return 0.60 if conditions else 0.40  # Expected conditions but didn't extract
        
        return 0.80  # No conditions needed or extracted
    
    def _score_sql_validity(self, sql: str) -> float:
        """Score SQL query validity"""
        sql_upper = sql.upper()
        
        # Basic SQL structure validation
        if not any(keyword in sql_upper for keyword in ['SELECT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
            return 0.20
        
        if 'FROM' not in sql_upper:
            return 0.30
        
        # Check for common SQL issues
        issues = 0
        
        # Check for dangerous operations
        if any(danger in sql_upper for danger in ['DROP', 'DELETE', 'INSERT', 'UPDATE']):
            issues += 1
        
        # Check for syntax patterns
        if sql.count('(') != sql.count(')'):
            issues += 1
        
        if sql.count("'") % 2 != 0:  # Unmatched quotes
            issues += 1
        
        base_score = 0.90
        return max(base_score - (issues * 0.15), 0.40)
    
    def _score_semantic_coherence(self, question: str, intent: str, mapped_columns: List[str]) -> float:
        """Score semantic coherence between question and generated query"""
        
        # Check if intent matches with mapped columns
        if intent in ['SUM', 'AVG', 'MAX', 'MIN'] and mapped_columns:
            # Should have numeric columns for math operations
            numeric_cols = [col for col in mapped_columns 
                          if self.schema_info['columns'][col]['type'] in ['INTEGER', 'REAL']]
            if numeric_cols:
                return 0.90
            else:
                return 0.50  # Math operation on non-numeric column
        
        if intent == 'COUNT' and mapped_columns:
            # Count operations can work on any column type
            return 0.85
        
        return 0.75  # Default reasonable coherence
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level category"""
        if confidence >= 0.85:
            return "HIGH"
        elif confidence >= 0.70:
            return "MEDIUM"
        elif confidence >= 0.55:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _generate_confidence_feedback(self, scores: Dict[str, float], overall: float) -> str:
        """Generate human-readable confidence feedback"""
        
        if overall >= 0.85:
            return "Excellent query understanding! The system is very confident about this interpretation."
        
        elif overall >= 0.70:
            return "Good query understanding. The system is reasonably confident about this interpretation."
        
        elif overall >= 0.55:
            return "Moderate query understanding. The interpretation may not be perfect - please verify the results."
        
        else:
            # Identify the weakest component
            weakest = min(scores.keys(), key=lambda k: scores[k])
            return f"Low confidence query interpretation. Issue may be with {weakest.replace('_', ' ')}. Consider rephrasing your question."
    
    def _get_improvement_suggestions(self, scores: Dict[str, float], question: str) -> List[str]:
        """Generate suggestions to improve query accuracy"""
        suggestions = []
        
        if scores['column_mapping_confidence'] < 0.70:
            available_cols = list(self.schema_info['columns'].keys())
            suggestions.append(f"Try using exact column names: {', '.join(available_cols[:5])}")
        
        if scores['intent_confidence'] < 0.70:
            suggestions.append("Be more specific about what you want: 'count', 'average', 'maximum', 'show', etc.")
        
        if scores['condition_extraction_confidence'] < 0.70 and any(c in question.lower() for c in ['>', '<', 'greater', 'less']):
            suggestions.append("Use clear comparison operators: 'greater than 100', 'less than 50', etc.")
        
        if scores['semantic_coherence_confidence'] < 0.70:
            suggestions.append("Make sure your question makes logical sense with your data types.")
        
        if not suggestions:
            suggestions.append("Query looks good! Results should be accurate.")
        
        return suggestions


# Enhanced Query Validator
class QueryValidator:
    """
    Validates generated queries against data constraints
    """
    
    def __init__(self, schema_info: Dict[str, Any]):
        self.schema_info = schema_info
    
    def validate_query_logic(self, sql: str, intent: str, mapped_columns: List[str]) -> Dict[str, Any]:
        """Validate query logic against data constraints"""
        
        issues = []
        warnings = []
        suggestions = []
        
        # Check column existence
        for col in mapped_columns:
            if col not in self.schema_info['columns']:
                issues.append(f"Column '{col}' does not exist in the data")
        
        # Check data type compatibility
        if intent in ['SUM', 'AVG', 'MAX', 'MIN']:
            numeric_cols = [col for col in mapped_columns 
                          if self.schema_info['columns'][col]['type'] in ['INTEGER', 'REAL']]
            if not numeric_cols:
                issues.append(f"Cannot perform {intent} operation on non-numeric columns")
        
        # Check for potential performance issues
        if 'LIMIT' not in sql.upper() and 'SELECT *' in sql.upper():
            warnings.append("Query may return large result set")
            suggestions.append("Consider adding specific column names or LIMIT clause")
        
        # Validate GROUP BY logic
        if 'GROUP BY' in sql.upper():
            if intent != 'COUNT' and intent not in ['SUM', 'AVG', 'MAX', 'MIN']:
                warnings.append("GROUP BY used with SELECT - results may be unexpected")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'suggestions': suggestions
        }