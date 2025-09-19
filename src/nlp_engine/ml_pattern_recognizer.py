# src/nlp_engine/ml_pattern_recognizer.py

import json
import re
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import pickle
import os

class MLQueryPatternRecognizer:
    """
    Machine Learning-based query pattern recognition for improved accuracy
    """
    
    def __init__(self, schema_info: Dict[str, Any]):
        self.schema_info = schema_info
        self.patterns_db = defaultdict(list)
        self.success_patterns = []
        self.failure_patterns = []
        
        # Load pre-trained patterns if available
        self.load_patterns()
        
        # Common successful query patterns
        self.initialize_base_patterns()
    
    def initialize_base_patterns(self):
        """Initialize with common successful query patterns"""
        
        base_patterns = [
            # COUNT patterns
            {
                'pattern': r'\bcount\s+(\w+)\s+by\s+(\w+)\b',
                'intent': 'COUNT',
                'template': 'SELECT {group_col}, COUNT(*) FROM data_table GROUP BY {group_col}',
                'confidence': 0.95,
                'example': 'count customers by gender'
            },
            
            # AVERAGE patterns
            {
                'pattern': r'\bwhat\s+is\s+the\s+average\s+(\w+)\b',
                'intent': 'AVG',
                'template': 'SELECT AVG({agg_col}) FROM data_table',
                'confidence': 0.90,
                'example': 'what is the average price'
            },
            
            # MAXIMUM patterns
            {
                'pattern': r'\bwhat\s+is\s+the\s+(?:maximum|highest|max)\s+(\w+)\b',
                'intent': 'MAX',
                'template': 'SELECT MAX({agg_col}) FROM data_table',
                'confidence': 0.90,
                'example': 'what is the maximum price'
            },
            
            # FILTER patterns
            {
                'pattern': r'\bshow\s+(\w+)\s+where\s+(\w+)\s*([><=]+)\s*(\d+(?:\.\d+)?)\b',
                'intent': 'SELECT',
                'template': 'SELECT * FROM data_table WHERE {filter_col} {operator} {value}',
                'confidence': 0.85,
                'example': 'show customers where age > 30'
            },
            
            # GROUP BY AVG patterns
            {
                'pattern': r'\baverage\s+(\w+)\s+by\s+(\w+)\b',
                'intent': 'AVG',
                'template': 'SELECT {group_col}, AVG({agg_col}) FROM data_table GROUP BY {group_col}',
                'confidence': 0.88,
                'example': 'average price by category'
            }
        ]
        
        self.success_patterns.extend(base_patterns)
    
    def recognize_query_pattern(self, question: str) -> Dict[str, Any]:
        """Recognize query pattern using ML-based approach"""
        
        question_lower = question.lower().strip()
        
        # Try to match against successful patterns
        best_match = self._find_best_pattern_match(question_lower)
        
        if best_match:
            return self._generate_query_from_pattern(question_lower, best_match)
        
        # Fallback to rule-based recognition
        return self._fallback_pattern_recognition(question_lower)
    
    def _find_best_pattern_match(self, question: str) -> Dict[str, Any]:
        """Find best matching pattern from learned patterns"""
        
        best_score = 0.0
        best_pattern = None
        
        for pattern_info in self.success_patterns:
            match = re.search(pattern_info['pattern'], question)
            if match:
                # Calculate confidence based on pattern success rate
                confidence = pattern_info['confidence']
                
                # Boost confidence if exact column names are found
                groups = match.groups()
                column_boost = self._calculate_column_matching_boost(groups)
                
                total_score = confidence + column_boost
                
                if total_score > best_score:
                    best_score = total_score
                    best_pattern = {
                        **pattern_info,
                        'match_groups': groups,
                        'final_confidence': min(total_score, 1.0)
                    }
        
        return best_pattern
    
    def _calculate_column_matching_boost(self, groups: Tuple) -> float:
        """Calculate confidence boost based on column name matching"""
        boost = 0.0
        
        for group in groups:
            if group and isinstance(group, str):
                # Check if group matches any column name
                for col_name in self.schema_info['columns'].keys():
                    if group.lower() == col_name.lower():
                        boost += 0.05
                    elif group.lower() in col_name.lower() or col_name.lower() in group.lower():
                        boost += 0.03
        
        return min(boost, 0.15)  # Cap the boost
    
    def _generate_query_from_pattern(self, question: str, pattern_match: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL query from matched pattern"""
        
        template = pattern_match['template']
        groups = pattern_match['match_groups']
        intent = pattern_match['intent']
        
        # Map groups to actual column names
        mapped_columns = self._map_groups_to_columns(groups)
        
        # Generate SQL based on template
        sql = self._fill_template(template, mapped_columns, groups)
        
        return {
            'sql': sql,
            'intent': intent,
            'confidence': pattern_match['final_confidence'],
            'pattern_used': pattern_match['pattern'],
            'mapped_columns': list(mapped_columns.values()),
            'method': 'pattern_matching'
        }
    
    def _map_groups_to_columns(self, groups: Tuple) -> Dict[str, str]:
        """Map regex groups to actual column names"""
        from fuzzywuzzy import fuzz
        
        mapped = {}
        column_names = list(self.schema_info['columns'].keys())
        
        for i, group in enumerate(groups):
            if group and isinstance(group, str):
                best_match = None
                best_score = 0
                
                for col_name in column_names:
                    # Exact match
                    if group.lower() == col_name.lower():
                        best_match = col_name
                        break
                    
                    # Fuzzy match
                    score = fuzz.ratio(group.lower(), col_name.lower())
                    if score > best_score and score > 70:
                        best_score = score
                        best_match = col_name
                
                if best_match:
                    if i == 0:
                        mapped['group_col'] = best_match
                        mapped['agg_col'] = best_match
                        mapped['filter_col'] = best_match
                    elif i == 1:
                        mapped['agg_col' if 'group_col' in mapped else 'group_col'] = best_match
                    
        return mapped
    
    def _fill_template(self, template: str, mapped_columns: Dict[str, str], groups: Tuple) -> str:
        """Fill SQL template with actual values"""
        
        sql = template
        
        # Replace column placeholders
        for placeholder, column in mapped_columns.items():
            sql = sql.replace(f'{{{placeholder}}}', column)
        
        # Replace operator and value placeholders
        if len(groups) >= 4:  # Filter pattern with operator and value
            operator = groups[2] if len(groups) > 2 else '='
            value = groups[3] if len(groups) > 3 else '0'
            
            sql = sql.replace('{operator}', operator)
            sql = sql.replace('{value}', str(value))
        
        return sql
    
    def _fallback_pattern_recognition(self, question: str) -> Dict[str, Any]:
        """Fallback pattern recognition using heuristics"""
        
        # Simple heuristic-based recognition
        if 'count' in question and 'by' in question:
            return {
                'intent': 'COUNT',
                'confidence': 0.70,
                'method': 'heuristic_fallback',
                'needs_groupby': True
            }
        
        if any(word in question for word in ['average', 'mean']):
            return {
                'intent': 'AVG',
                'confidence': 0.65,
                'method': 'heuristic_fallback'
            }
        
        if any(word in question for word in ['maximum', 'max', 'highest']):
            return {
                'intent': 'MAX',
                'confidence': 0.65,
                'method': 'heuristic_fallback'
            }
        
        return {
            'intent': 'SELECT',
            'confidence': 0.50,
            'method': 'default_fallback'
        }
    
    def learn_from_feedback(self, question: str, generated_sql: str, user_feedback: str, actual_intent: str = None):
        """Learn from user feedback to improve future predictions"""
        
        pattern_data = {
            'question': question.lower().strip(),
            'sql': generated_sql,
            'feedback': user_feedback,  # 'correct', 'incorrect', 'partially_correct'
            'actual_intent': actual_intent,
            'timestamp': self._get_timestamp()
        }
        
        if user_feedback == 'correct':
            self._add_successful_pattern(pattern_data)
        elif user_feedback == 'incorrect':
            self._add_failed_pattern(pattern_data)
        
        # Save updated patterns
        self.save_patterns()
    
    def _add_successful_pattern(self, pattern_data: Dict[str, Any]):
        """Add successful pattern to learning database"""
        
        # Extract pattern from successful query
        question = pattern_data['question']
        
        # Try to generalize the pattern
        generalized_pattern = self._generalize_pattern(question)
        
        if generalized_pattern:
            self.success_patterns.append({
                'pattern': generalized_pattern,
                'intent': pattern_data.get('actual_intent', 'SELECT'),
                'confidence': 0.80,  # Start with medium confidence
                'learned': True,
                'success_count': 1
            })
    
    def _generalize_pattern(self, question: str) -> str:
        """Generalize a successful question into a regex pattern"""
        
        # Replace specific column names with placeholders
        generalized = question
        
        for col_name in self.schema_info['columns'].keys():
            if col_name.lower() in question:
                generalized = generalized.replace(col_name.lower(), r'(\w+)')
        
        # Replace numbers with number patterns
        generalized = re.sub(r'\b\d+(?:\.\d+)?\b', r'(\d+(?:\.\d+)?)', generalized)
        
        # Escape special regex characters
        generalized = re.escape(generalized)
        
        # Unescape the patterns we want to keep
        generalized = generalized.replace(r'\(\w\+\)', r'(\w+)')
        generalized = generalized.replace(r'\(\d\+\(\?\:\\\.\d\+\)\?\)', r'(\d+(?:\.\d+)?)')
        
        return generalized
    
    def _add_failed_pattern(self, pattern_data: Dict[str, Any]):
        """Add failed pattern to avoid similar mistakes"""
        self.failure_patterns.append(pattern_data)
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned patterns"""
        
        return {
            'total_success_patterns': len(self.success_patterns),
            'total_failure_patterns': len(self.failure_patterns),
            'learned_patterns': len([p for p in self.success_patterns if p.get('learned', False)]),
            'base_patterns': len([p for p in self.success_patterns if not p.get('learned', False)])
        }
    
    def save_patterns(self):
        """Save learned patterns to disk"""
        try:
            patterns_data = {
                'success_patterns': self.success_patterns,
                'failure_patterns': self.failure_patterns
            }
            
            os.makedirs('data/learned_patterns', exist_ok=True)
            with open('data/learned_patterns/query_patterns.json', 'w') as f:
                json.dump(patterns_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save patterns: {e}")
    
    def load_patterns(self):
        """Load previously learned patterns"""
        try:
            if os.path.exists('data/learned_patterns/query_patterns.json'):
                with open('data/learned_patterns/query_patterns.json', 'r') as f:
                    patterns_data = json.load(f)
                
                self.success_patterns = patterns_data.get('success_patterns', [])
                self.failure_patterns = patterns_data.get('failure_patterns', [])
        except Exception as e:
            print(f"Warning: Could not load patterns: {e}")
    
    def _get_timestamp(self):
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()


# Enhanced Query Accuracy Optimizer
class QueryAccuracyOptimizer:
    """
    Combines all accuracy improvement techniques
    """
    
    def __init__(self, schema_info: Dict[str, Any]):
        self.schema_info = schema_info
        
        # Initialize all components
        try:
            self.semantic_mapper = SemanticColumnMapper(schema_info)
            self.confidence_scorer = QueryConfidenceScorer(schema_info)
            self.pattern_recognizer = MLQueryPatternRecognizer(schema_info)
            self.query_validator = QueryValidator(schema_info)
            self.advanced_text_processor = AdvancedTextProcessor()
        except Exception as e:
            print(f"Warning: Some advanced features may not be available: {e}")
            self.semantic_mapper = None
    
    def optimize_query_generation(self, question: str) -> Dict[str, Any]:
        """
        Main method that combines all accuracy improvement techniques
        """
        
        # Step 1: Advanced pattern recognition
        pattern_result = self.pattern_recognizer.recognize_query_pattern(question)
        
        # Step 2: Enhanced intent recognition
        intent, intent_confidence = self.advanced_text_processor.extract_intent_with_confidence(question)
        
        # Step 3: Semantic column mapping
        if self.semantic_mapper:
            column_matches = self.semantic_mapper.find_best_columns(question, intent)
            best_columns = [col for col, score in column_matches if score > 0.6]
        else:
            best_columns = []
        
        # Step 4: Advanced condition extraction
        if self.semantic_mapper and best_columns:
            conditions = self.semantic_mapper.extract_value_constraints(question, best_columns)
        else:
            conditions = []
        
        # Step 5: Generate optimized result
        optimized_result = {
            'intent': intent,
            'confidence': intent_confidence,
            'mapped_columns': best_columns,
            'conditions': conditions,
            'pattern_match': pattern_result,
            'groupby_needed': self.advanced_text_processor.extract_groupby_intent(question),
            'optimization_applied': True
        }
        
        return optimized_result
    
    def validate_and_score_query(self, question: str, sql: str, intent: str, columns: List[str], conditions: List[Dict]) -> Dict[str, Any]:
        """
        Validate and score the generated query
        """
        
        # Calculate comprehensive confidence
        confidence_result = self.confidence_scorer.calculate_comprehensive_confidence(
            question, intent, columns, conditions, sql
        )
        
        # Validate query logic
        validation_result = self.query_validator.validate_query_logic(sql, intent, columns)
        
        return {
            'confidence_analysis': confidence_result,
            'validation_result': validation_result,
            'overall_quality': self._calculate_overall_quality(confidence_result, validation_result)
        }
    
    def _calculate_overall_quality(self, confidence_result: Dict, validation_result: Dict) -> str:
        """Calculate overall query quality"""
        
        if not validation_result['is_valid']:
            return "POOR"
        
        confidence_level = confidence_result['confidence_level']
        has_warnings = len(validation_result['warnings']) > 0
        
        if confidence_level == "HIGH" and not has_warnings:
            return "EXCELLENT"
        elif confidence_level in ["HIGH", "MEDIUM"] and not has_warnings:
            return "GOOD"
        elif confidence_level == "MEDIUM" or (confidence_level == "HIGH" and has_warnings):
            return "FAIR"
        else:
            return "POOR"