# src/data_handler/csv_processor.py

import pandas as pd
import numpy as np
import sqlite3
import chardet
from typing import Dict, List, Optional, Any
import os
import json

class CSVProcessor:
    def __init__(self):
        self.df = None
        self.schema = None
        self.file_info = None
        self.table_name = "data_table"
    
    def load_csv(self, file_path: str) -> Dict[str, Any]:
        try:
            # Simple CSV loading
            self.df = pd.read_csv(file_path)
            
            # Clean column names
            self.df.columns = self.df.columns.str.strip()
            
            # Generate schema
            self.schema = self._detect_schema()
            
            # Store file information
            self.file_info = {
                'filename': os.path.basename(file_path),
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'size_mb': round(os.path.getsize(file_path) / (1024*1024), 2) if os.path.exists(file_path) else 0
            }
            
            return {
                'success': True,
                'message': f"Successfully loaded {self.file_info['rows']} rows and {self.file_info['columns']} columns",
                'file_info': self.file_info
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Error loading CSV: {str(e)}",
                'file_info': None
            }
    
    def _detect_schema(self) -> Dict[str, Any]:
        if self.df is None:
            return {}
        
        schema = {}
        
        for column in self.df.columns:
            col_data = self.df[column].dropna()
            
            schema[column] = {
                'type': self._detect_column_type(col_data),
                'nullable': self.df[column].isnull().any(),
                'null_count': self.df[column].isnull().sum(),
                'unique_count': self.df[column].nunique(),
                'sample_values': col_data.head(3).tolist() if len(col_data) > 0 else [],
                'is_primary_key': self._is_potential_primary_key(column)
            }
        
        return schema
    
    def _detect_column_type(self, series: pd.Series) -> str:
        if len(series) == 0:
            return 'TEXT'
        
        # Try numeric
        try:
            pd.to_numeric(series)
            return 'INTEGER' if all(float(x).is_integer() for x in series if pd.notna(x)) else 'REAL'
        except:
            pass
        
        # Try datetime
        try:
            pd.to_datetime(series)
            return 'DATETIME'
        except:
            pass
        
        return 'TEXT'
    
    def _is_potential_primary_key(self, column: str) -> bool:
        if self.df is None:
            return False
        
        col_data = self.df[column]
        return col_data.nunique() == len(col_data) and not col_data.isnull().any()
    
    def get_preview(self, rows: int = 5) -> pd.DataFrame:
        if self.df is None:
            return pd.DataFrame()
        return self.df.head(rows)
    
    def to_sqlite(self, db_path: str) -> bool:
        try:
            if self.df is None:
                return False
            
            conn = sqlite3.connect(db_path)
            self.df.to_sql(self.table_name, conn, if_exists='replace', index=False)
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error converting to SQLite: {e}")
            return False
    
    def get_schema_info(self) -> Dict[str, Any]:
        if self.schema is None:
            return {}
        
        return {
            'columns': self.schema,
            'table_name': self.table_name,
            'total_rows': len(self.df) if self.df is not None else 0,
            'column_names': list(self.schema.keys()),
            'numeric_columns': [col for col, info in self.schema.items() if info['type'] in ['INTEGER', 'REAL']],
            'text_columns': [col for col, info in self.schema.items() if info['type'] == 'TEXT'],
            'date_columns': [col for col, info in self.schema.items() if info['type'] == 'DATETIME']
        }
    
    def validate_data_quality(self) -> Dict[str, Any]:
        if self.df is None:
            return {}
        
        return {
            'duplicate_rows': self.df.duplicated().sum(),
            'empty_columns': [col for col in self.df.columns if self.df[col].isnull().all()],
            'columns_with_nulls': []
        }