# src/sql_generator/query_builder.py

from typing import Dict, List, Optional, Any
import re

class QueryBuilder:
    """
    Enhanced query builder with proper GROUP BY and column selection
    """
    
    def __init__(self, schema_info: Dict[str, Any]):
        self.schema_info = schema_info
        self.table_name = schema_info.get('table_name', 'data_table')
        self.columns = schema_info.get('columns', {})
        self.column_names = list(self.columns.keys())
        self.numeric_columns = schema_info.get('numeric_columns', [])
    
    def build_query(self, processed_query: Dict[str, Any], mapped_columns: List[str], 
                   conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhanced query building with GROUP BY support
        """
        
        intent = processed_query.get('intent', 'SELECT')
        original_query = processed_query.get('original_query', '').lower()
        
        try:
            # Check if this is a GROUP BY query
            is_groupby_query = self._is_groupby_query(original_query, intent, mapped_columns)
            
            if intent == 'SELECT':
                sql = self._build_select_query(mapped_columns, conditions)
            elif intent == 'COUNT' and is_groupby_query:
                sql = self._build_groupby_count_query(mapped_columns, conditions)
            elif intent == 'COUNT':
                sql = self._build_simple_count_query(mapped_columns, conditions)
            elif intent in ['SUM', 'AVG', 'MAX', 'MIN']:
                if is_groupby_query:
                    sql = self._build_groupby_aggregation_query(intent, mapped_columns, conditions)
                else:
                    sql = self._build_simple_aggregation_query(intent, mapped_columns, conditions)
            else:
                sql = self._build_default_query()
            
            return {
                'sql': sql,
                'intent': intent,
                'success': True,
                'explanation': self._explain_query(sql, intent, is_groupby_query),
                'columns_used': mapped_columns,
                'conditions_used': conditions
            }
            
        except Exception as e:
            return {
                'sql': self._build_default_query(),
                'intent': intent,
                'success': False,
                'error': str(e),
                'explanation': "Generated a simple query due to processing error"
            }
    
    def _is_groupby_query(self, original_query: str, intent: str, mapped_columns: List[str]) -> bool:
        """Detect if this should be a GROUP BY query"""
        
        # Keywords that indicate grouping
        groupby_keywords = ['by', 'per', 'each', 'for each', 'in each']
        
        # Check for groupby patterns
        for keyword in groupby_keywords:
            if keyword in original_query:
                return True
        
        # If intent is COUNT/SUM/AVG and we have mapped columns, likely a group by
        if intent in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN'] and mapped_columns:
            # Check if the column is likely categorical (for grouping)
            for col in mapped_columns:
                if self._is_categorical_column(col):
                    return True
        
        return False
    
    def _is_categorical_column(self, column: str) -> bool:
        """Check if a column is likely categorical (good for grouping)"""
        col_lower = column.lower()
        
        # Common categorical column patterns
        categorical_patterns = [
            'gender', 'semester', 'department', 'grade', 'class', 'category',
            'type', 'status', 'level', 'group', 'team', 'division', 'mall'
        ]
        
        for pattern in categorical_patterns:
            if pattern in col_lower:
                return True
        
        # Check if column has low unique values (likely categorical)
        if column in self.columns:
            unique_count = self.columns[column].get('unique_count', 0)
            total_rows = self.schema_info.get('total_rows', 1)
            if unique_count < total_rows * 0.1:  # Less than 10% unique values
                return True
        
        return False
    
    def _build_groupby_count_query(self, mapped_columns: List[str], conditions: List[Dict[str, Any]]) -> str:
        """Build COUNT query with GROUP BY"""
        
        if not mapped_columns:
            return self._build_simple_count_query([], conditions)
        
        group_column = mapped_columns[0]  # Use first mapped column for grouping
        
        # Build query
        select_clause = f"SELECT {group_column}, COUNT(*) as count"
        from_clause = f"FROM {self.table_name}"
        group_clause = f"GROUP BY {group_column}"
        order_clause = f"ORDER BY count DESC"
        
        # WHERE clause
        where_clause = self._build_where_clause(conditions)
        
        # Combine parts
        query = f"{select_clause} {from_clause}"
        if where_clause:
            query += f" {where_clause}"
        query += f" {group_clause} {order_clause}"
        
        return query
    
    def _build_simple_count_query(self, mapped_columns: List[str], conditions: List[Dict[str, Any]]) -> str:
        """Build simple COUNT query"""
        
        count_clause = "SELECT COUNT(*)"
        from_clause = f"FROM {self.table_name}"
        where_clause = self._build_where_clause(conditions)
        
        query = f"{count_clause} {from_clause}"
        if where_clause:
            query += f" {where_clause}"
        
        return query
    
    def _build_groupby_aggregation_query(self, intent: str, mapped_columns: List[str], 
                                       conditions: List[Dict[str, Any]]) -> str:
        """Build aggregation query with GROUP BY"""
        
        if len(mapped_columns) < 2:
            return self._build_simple_aggregation_query(intent, mapped_columns, conditions)
        
        # First column for grouping, second for aggregation
        group_column = mapped_columns[0]
        agg_column = self._find_best_aggregation_column(intent, mapped_columns[1:])
        
        if not agg_column:
            agg_column = mapped_columns[1]
        
        # Build query
        select_clause = f"SELECT {group_column}, {intent}({agg_column}) as {intent.lower()}_{agg_column}"
        from_clause = f"FROM {self.table_name}"
        group_clause = f"GROUP BY {group_column}"
        order_clause = f"ORDER BY {intent.lower()}_{agg_column} DESC"
        
        # WHERE clause
        where_clause = self._build_where_clause(conditions)
        
        # Combine parts
        query = f"{select_clause} {from_clause}"
        if where_clause:
            query += f" {where_clause}"
        query += f" {group_clause} {order_clause}"
        
        return query
    
    def _build_simple_aggregation_query(self, intent: str, mapped_columns: List[str], 
                                      conditions: List[Dict[str, Any]]) -> str:
        """Build simple aggregation query with enhanced column selection"""
        
        agg_column = self._find_best_aggregation_column(intent, mapped_columns)
        
        if not agg_column:
            agg_column = self.numeric_columns[0] if self.numeric_columns else '*'
        
        agg_clause = f"SELECT {intent}({agg_column})"
        from_clause = f"FROM {self.table_name}"
        where_clause = self._build_where_clause(conditions)
        
        query = f"{agg_clause} {from_clause}"
        if where_clause:
            query += f" {where_clause}"
        
        return query
    
    def _find_best_aggregation_column(self, intent: str, mapped_columns: List[str]) -> Optional[str]:
        """Enhanced logic to find the BEST column for aggregation"""
        
        if not mapped_columns:
            return self._find_default_aggregation_column()
        
        # Priority 1: Look for value-related columns in mapped columns
        value_keywords = ['price', 'cost', 'amount', 'salary', 'revenue', 'sales', 
                          'score', 'marks', 'rating', 'value', 'total', 'quantity']
        
        # Check mapped columns for value keywords
        for col in mapped_columns:
            if col in self.numeric_columns:
                col_lower = col.lower()
                for keyword in value_keywords:
                    if keyword in col_lower:
                        return col
        
        # Priority 2: Return first numeric column from mapped columns
        for col in mapped_columns:
            if col in self.numeric_columns:
                return col
        
        # Priority 3: Fallback to default
        return self._find_default_aggregation_column()
    
    def _find_default_aggregation_column(self) -> Optional[str]:
        """Find default aggregation column"""
        
        # Look for common aggregatable columns
        priority_columns = ['price', 'salary', 'amount', 'revenue', 'cost', 'score', 'marks']
        
        for priority in priority_columns:
            for col in self.numeric_columns:
                if priority.lower() in col.lower():
                    return col
        
        # Return first numeric column
        return self.numeric_columns[0] if self.numeric_columns else None
    
    def _build_select_query(self, mapped_columns: List[str], conditions: List[Dict[str, Any]]) -> str:
        """Build SELECT query"""
        
        if mapped_columns:
            select_clause = f"SELECT {', '.join(mapped_columns)}"
        else:
            select_clause = "SELECT *"
        
        from_clause = f"FROM {self.table_name}"
        where_clause = self._build_where_clause(conditions)
        
        query = f"{select_clause} {from_clause}"
        if where_clause:
            query += f" {where_clause}"
        
        # Add LIMIT for safety
        query += " LIMIT 100"
        
        return query
    
    def _build_where_clause(self, conditions: List[Dict[str, Any]]) -> str:
        """Build WHERE clause from conditions"""
        
        if not conditions:
            return ""
        
        where_parts = []
        
        for condition in conditions:
            column = condition['column']
            operator = condition['operator']
            value = condition['value']
            col_type = condition.get('type', 'TEXT')
            
            # Format value based on type
            if col_type == 'TEXT':
                formatted_value = f"'{value}'"
            else:
                formatted_value = str(value)
            
            where_parts.append(f"{column} {operator} {formatted_value}")
        
        return f"WHERE {' AND '.join(where_parts)}"
    
    def _build_default_query(self) -> str:
        """Build safe default query"""
        return f"SELECT * FROM {self.table_name} LIMIT 10"
    
    def _explain_query(self, sql: str, intent: str, is_groupby: bool) -> str:
        """Generate human-readable explanation"""
        
        explanations = {
            'SELECT': "This query shows the requested data from your table.",
            'COUNT': "This query counts records" + (" grouped by category." if is_groupby else "."),
            'SUM': "This query calculates the total sum" + (" by group." if is_groupby else "."),
            'AVG': "This query calculates the average" + (" by group." if is_groupby else "."),
            'MAX': "This query finds the maximum value" + (" by group." if is_groupby else "."),
            'MIN': "This query finds the minimum value" + (" by group." if is_groupby else ".")
        }
        
        base_explanation = explanations.get(intent, "This query retrieves data based on your request.")
        
        if "WHERE" in sql:
            base_explanation += " It includes filters to show only records that meet your conditions."
        
        if "LIMIT" in sql:
            base_explanation += " Results are limited for better performance."
        
        return base_explanation