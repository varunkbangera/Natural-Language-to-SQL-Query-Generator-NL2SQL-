# src/nlp_engine/semantic_mapper.py

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple, Any
import re

class SemanticColumnMapper:
    """
    Advanced column mapping using semantic embeddings for higher accuracy
    """
    
    def __init__(self, schema_info: Dict[str, Any]):
        self.schema_info = schema_info
        self.columns = list(schema_info['columns'].keys())
        
        # Load lightweight sentence transformer model
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_embeddings = True
        except:
            print("Warning: Sentence transformers not available, using fallback method")
            self.use_embeddings = False
        
        # Pre-compute column embeddings
        if self.use_embeddings:
            self.column_embeddings = self._compute_column_embeddings()
        
        # Enhanced column synonyms with domain knowledge
        self.domain_synonyms = self._build_domain_synonyms()
    
    def _compute_column_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for all columns"""
        embeddings = {}
        
        for col in self.columns:
            # Create rich description for each column
            col_description = self._create_column_description(col)
            embeddings[col] = self.model.encode(col_description)
        
        return embeddings
    
    def _create_column_description(self, column: str) -> str:
        """Create rich description for column based on name and data"""
        col_info = self.schema_info['columns'][column]
        
        # Base description
        description = f"{column} "
        
        # Add type information
        if col_info['type'] == 'INTEGER':
            description += "number integer count "
        elif col_info['type'] == 'REAL':
            description += "number decimal amount value price cost "
        elif col_info['type'] == 'TEXT':
            description += "text name category label "
        
        # Add domain-specific context based on column name
        col_lower = column.lower()
        
        # Customer/User related
        if any(word in col_lower for word in ['customer', 'user', 'person', 'client']):
            description += "customer user person client individual "
        
        # Financial
        if any(word in col_lower for word in ['price', 'cost', 'amount', 'revenue', 'salary']):
            description += "money financial currency payment cost expense revenue income "
        
        # Demographics
        if 'age' in col_lower:
            description += "age years old demographic "
        if 'gender' in col_lower:
            description += "gender sex male female demographic "
        
        # Categories
        if any(word in col_lower for word in ['category', 'type', 'class', 'group']):
            description += "category classification type group class "
        
        # Quantities
        if any(word in col_lower for word in ['quantity', 'count', 'number']):
            description += "quantity amount count number how many "
        
        # Dates
        if any(word in col_lower for word in ['date', 'time', 'created', 'updated']):
            description += "date time when created timestamp "
        
        # Add sample values context for categorical columns
        if col_info.get('unique_count', 0) < 20 and col_info.get('sample_values'):
            sample_str = ' '.join(map(str, col_info['sample_values'][:5]))
            description += f"values {sample_str} "
        
        return description.strip()
    
    def _build_domain_synonyms(self) -> Dict[str, List[str]]:
        """Build domain-specific synonyms"""
        return {
            # Customer synonyms
            'customer': ['client', 'user', 'buyer', 'person', 'individual', 'people'],
            'customers': ['clients', 'users', 'buyers', 'people', 'individuals'],
            
            # Financial synonyms
            'price': ['cost', 'amount', 'value', 'money', 'fee', 'charge'],
            'revenue': ['sales', 'income', 'earnings', 'proceeds'],
            'salary': ['wage', 'pay', 'income', 'compensation'],
            
            # Quantity synonyms
            'quantity': ['amount', 'count', 'number', 'how many', 'total'],
            'count': ['number', 'total', 'quantity', 'how many'],
            
            # Demographics
            'age': ['years', 'old', 'age group'],
            'gender': ['sex', 'male female'],
            
            # Categories
            'category': ['type', 'class', 'group', 'classification', 'kind'],
            'type': ['category', 'class', 'group', 'kind'],
            
            # Location
            'location': ['place', 'address', 'where', 'position'],
            'mall': ['store', 'shop', 'center', 'location'],
            
            # Time
            'date': ['time', 'when', 'created', 'timestamp'],
            'time': ['date', 'when', 'timestamp']
        }
    
    def find_best_columns(self, question: str, intent: str) -> List[Tuple[str, float]]:
        """
        Find best matching columns using semantic similarity
        Returns list of (column_name, confidence_score) tuples
        """
        if not self.use_embeddings:
            return self._fallback_column_matching(question, intent)
        
        # Create query embedding
        query_embedding = self.model.encode(question)
        
        # Calculate similarities
        similarities = []
        for col, col_embedding in self.column_embeddings.items():
            similarity = np.dot(query_embedding, col_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(col_embedding)
            )
            similarities.append((col, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Apply intent-based filtering
        filtered_similarities = self._apply_intent_filtering(similarities, intent)
        
        return filtered_similarities[:3]  # Return top 3 matches
    
    def _apply_intent_filtering(self, similarities: List[Tuple[str, float]], intent: str) -> List[Tuple[str, float]]:
        """Apply intent-based filtering to improve accuracy"""
        
        if intent in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']:
            # For aggregations, boost numeric columns and categorical columns
            boosted = []
            for col, score in similarities:
                col_type = self.schema_info['columns'][col]['type']
                
                if intent == 'COUNT':
                    # For COUNT, prefer categorical columns
                    if col_type == 'TEXT' or self.schema_info['columns'][col].get('unique_count', 0) < 50:
                        score *= 1.2
                elif intent in ['SUM', 'AVG', 'MAX', 'MIN']:
                    # For math operations, prefer numeric columns
                    if col_type in ['INTEGER', 'REAL']:
                        score *= 1.3
                
                boosted.append((col, score))
            
            return sorted(boosted, key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def _fallback_column_matching(self, question: str, intent: str) -> List[Tuple[str, float]]:
        """Fallback method without embeddings"""
        from fuzzywuzzy import fuzz
        
        question_lower = question.lower()
        matches = []
        
        for col in self.columns:
            score = 0.0
            
            # Direct name match
            if col.lower() in question_lower:
                score = 0.9
            else:
                # Fuzzy match
                score = fuzz.ratio(col.lower(), question_lower) / 100.0
            
            # Apply synonym matching
            for word in question_lower.split():
                for synonym_key, synonyms in self.domain_synonyms.items():
                    if word in synonyms and synonym_key in col.lower():
                        score = max(score, 0.8)
            
            matches.append((col, score))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def extract_value_constraints(self, question: str, target_columns: List[str]) -> List[Dict[str, Any]]:
        """Extract value constraints with context awareness"""
        constraints = []
        
        # Enhanced regex patterns for operators
        operator_patterns = {
            'greater_than': [
                r'(?:greater than|more than|above|over|higher than|>\s*)',
                r'(?:exceeds?|beyond)'
            ],
            'less_than': [
                r'(?:less than|below|under|lower than|<\s*)',
                r'(?:fewer than|beneath)'
            ],
            'equal': [
                r'(?:equal(?:s)? to|is|=\s*)',
                r'(?:exactly|precisely)'
            ],
            'between': [
                r'between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)'
            ]
        }
        
        # Extract numeric values and their context
        import re
        
        for target_col in target_columns:
            col_info = self.schema_info['columns'][target_col]
            
            # Only process numeric columns for numeric constraints
            if col_info['type'] not in ['INTEGER', 'REAL']:
                continue
            
            for op_type, patterns in operator_patterns.items():
                for pattern in patterns:
                    if op_type == 'between':
                        # Special handling for BETWEEN
                        matches = re.finditer(pattern, question.lower())
                        for match in matches:
                            constraints.append({
                                'column': target_col,
                                'operator': 'BETWEEN',
                                'value': [float(match.group(1)), float(match.group(2))],
                                'confidence': 0.9
                            })
                    else:
                        # Regular operators
                        full_pattern = pattern + r'(\d+(?:\.\d+)?)'
                        matches = re.finditer(full_pattern, question.lower())
                        
                        for match in matches:
                            value = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
                            
                            # Context validation - check if value makes sense
                            if self._validate_constraint_value(target_col, value, op_type):
                                sql_op = self._convert_to_sql_operator(op_type)
                                constraints.append({
                                    'column': target_col,
                                    'operator': sql_op,
                                    'value': value,
                                    'confidence': 0.8
                                })
        
        return constraints
    
    def _validate_constraint_value(self, column: str, value: Any, operator: str) -> bool:
        """Validate if constraint value makes sense for the column"""
        col_info = self.schema_info['columns'][column]
        
        # Get column statistics if available
        if 'stats' in col_info:
            stats = col_info['stats']
            min_val = stats.get('min', 0)
            max_val = stats.get('max', float('inf'))
            
            # Check if value is within reasonable range
            if operator == 'greater_than' and value >= max_val:
                return False
            if operator == 'less_than' and value <= min_val:
                return False
        
        return True
    
    def _convert_to_sql_operator(self, op_type: str) -> str:
        """Convert operator type to SQL operator"""
        mapping = {
            'greater_than': '>',
            'less_than': '<',
            'equal': '=',
            'greater_equal': '>=',
            'less_equal': '<='
        }
        return mapping.get(op_type, '=')


# Enhanced Text Processor with better intent recognition
class AdvancedTextProcessor:
    """
    Advanced text processor with improved intent recognition
    """
    
    def __init__(self):
        # Enhanced intent patterns with more variations
        self.intent_patterns = {
            'SELECT': [
                r'\b(?:show|display|list|get|find|retrieve|fetch|give me)\b',
                r'\bwhat (?:are|is)\b',
                r'\btell me\b'
            ],
            'COUNT': [
                r'\b(?:how many|count|number of|total number)\b',
                r'\bcount\s+\w+\s+by\b',  # "count customers by gender"
                r'\bhow much\b'
            ],
            'SUM': [
                r'\b(?:sum|total|add up)\b',
                r'\btotal\s+\w+\b',
                r'\bsum of\b'
            ],
            'AVG': [
                r'\b(?:average|mean|avg)\b',
                r'\bwhat(?:\'s| is) the average\b'
            ],
            'MAX': [
                r'\b(?:maximum|max|highest|largest|biggest|top)\b',
                r'\bmost expensive\b',
                r'\bhighest\s+\w+\b'
            ],
            'MIN': [
                r'\b(?:minimum|min|lowest|smallest|cheapest)\b',
                r'\blowest\s+\w+\b'
            ]
        }
        
        # Enhanced comparison patterns
        self.comparison_patterns = {
            'greater_than': [
                r'\b(?:greater than|more than|above|over|higher than|exceeds?|beyond)\s*(\d+(?:\.\d+)?)\b',
                r'\b>\s*(\d+(?:\.\d+)?)\b',
                r'\b(\d+(?:\.\d+)?)\s*(?:\+|and above|or more)\b'
            ],
            'less_than': [
                r'\b(?:less than|below|under|lower than|fewer than|beneath)\s*(\d+(?:\.\d+)?)\b',
                r'\b<\s*(\d+(?:\.\d+)?)\b',
                r'\bunder\s*(\d+(?:\.\d+)?)\b'
            ],
            'equal': [
                r'\b(?:equal(?:s)? to|is|exactly|precisely)\s*(\d+(?:\.\d+)?)\b',
                r'\b=\s*(\d+(?:\.\d+)?)\b'
            ]
        }
    
    def extract_intent_with_confidence(self, query: str) -> Tuple[str, float]:
        """Extract intent with confidence score"""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    # Calculate confidence based on pattern specificity
                    confidence = 0.9 if len(pattern) > 20 else 0.7
                    return intent, confidence
        
        return 'SELECT', 0.5
    
    def extract_groupby_intent(self, query: str) -> bool:
        """Detect if query needs GROUP BY"""
        groupby_indicators = [
            r'\bby\s+\w+',
            r'\bper\s+\w+',
            r'\beach\s+\w+',
            r'\bfor each\s+\w+',
            r'\bin each\s+\w+'
        ]
        
        query_lower = query.lower()
        for pattern in groupby_indicators:
            if re.search(pattern, query_lower):
                return True
        
        return False