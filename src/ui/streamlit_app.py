# src/ui/streamlit_app.py

import streamlit as st
import pandas as pd
import sqlite3
import sys
import os
from typing import Dict, List, Any

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_handler.csv_processor import CSVProcessor
from nlp_engine.text_processor import TextProcessor
from nlp_engine.schema_mapper import SchemaMapper
from sql_generator.query_builder import QueryBuilder

# Page setup
st.set_page_config(page_title="NL2SQL", page_icon="🤖", layout="wide")

class HighAccuracyNL2SQLApp:
    def __init__(self):
        self.csv_processor = CSVProcessor()
        self.text_processor = TextProcessor()
        self.schema_mapper = None
        self.query_builder = None
        
        # Session state
        if 'page' not in st.session_state:
            st.session_state.page = "upload"
        if 'data_ready' not in st.session_state:
            st.session_state.data_ready = False
        if 'detailed_schema' not in st.session_state:
            st.session_state.detailed_schema = None
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
    
    def run(self):
        if st.session_state.page == "upload":
            self.upload_page()
        else:
            self.query_page()
    
    def upload_page(self):
        """Enhanced upload page with detailed data analysis"""
        
        st.title("🤖 High-Accuracy Data Query Assistant")
        st.write("Upload your CSV file for intelligent data analysis and querying")
        
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader("📁 Choose CSV file", type=['csv'])
        
        if uploaded_file:
            self.process_file_enhanced(uploaded_file)
        
        st.markdown("### Or try sample data:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👥 Employees", key="btn_employees"):
                self.load_sample("data/sample_datasets/employees.csv")
        with col2:
            if st.button("💰 Sales", key="btn_sales"):
                self.load_sample("data/sample_datasets/sales.csv") 
        with col3:
            if st.button("🎓 Students", key="btn_students"):
                self.load_sample("data/sample_datasets/students.csv")
    
    def process_file_enhanced(self, uploaded_file):
        """Enhanced file processing with detailed analysis"""
        
        # Save file
        with open("temp.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process with detailed analysis
        with st.spinner("🔍 Performing detailed data analysis..."):
            result = self.csv_processor.load_csv("temp.csv")
        
        if result['success']:
            # Basic info
            st.success(f"✅ Successfully loaded {result['file_info']['rows']:,} rows!")
            
            # Get enhanced schema
            detailed_schema = self.get_detailed_schema_analysis()
            st.session_state.detailed_schema = detailed_schema
            
            # Show comprehensive data overview
            self.show_comprehensive_overview(detailed_schema)
            
            # Show data preview
            st.subheader("👀 Data Preview")
            preview = self.csv_processor.get_preview(10)  # Show more rows
            st.dataframe(preview, use_container_width=True)
            
            # Show smart query suggestions
            self.show_intelligent_suggestions(detailed_schema)
            
            # Start button
            if st.button("🚀 Start Intelligent Querying", type="primary", key="start_smart_queries"):
                self.setup_enhanced_queries()
                
        else:
            st.error(f"❌ Error: {result['message']}")
    
    def get_detailed_schema_analysis(self):
        """Perform detailed schema analysis"""
        df = self.csv_processor.df
        schema = self.csv_processor.get_schema_info()
        
        detailed_schema = {
            'basic_schema': schema,
            'column_details': {}
        }
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            # Get unique values (limited to first 20 for display)
            unique_values = col_data.unique()
            unique_count = len(unique_values)
            
            # Determine if it's categorical based on unique values
            is_categorical = unique_count <= 50 or unique_count < len(df) * 0.05
            
            # Get sample values
            if is_categorical and unique_count <= 20:
                sample_values = list(unique_values)
            else:
                sample_values = list(col_data.head(10))
            
            # Statistical analysis for numeric columns
            stats = {}
            if col in schema['numeric_columns']:
                stats = {
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median())
                }
            
            detailed_schema['column_details'][col] = {
                'type': schema['columns'][col]['type'],
                'unique_count': unique_count,
                'is_categorical': is_categorical,
                'sample_values': sample_values,
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'stats': stats
            }
        
        return detailed_schema
    
    def show_comprehensive_overview(self, detailed_schema):
        """Show comprehensive data overview"""
        
        st.subheader("📊 Comprehensive Data Analysis")
        
        # Data quality metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_rows = detailed_schema['basic_schema']['total_rows']
        total_cols = len(detailed_schema['column_details'])
        numeric_cols = len(detailed_schema['basic_schema']['numeric_columns'])
        text_cols = len(detailed_schema['basic_schema']['text_columns'])
        
        with col1:
            st.metric("📊 Total Rows", f"{total_rows:,}")
        with col2:
            st.metric("📋 Total Columns", total_cols)
        with col3:
            st.metric("🔢 Numeric Columns", numeric_cols)
        with col4:
            st.metric("📝 Text Columns", text_cols)
        
        # Detailed column analysis
        st.subheader("🔍 Column Analysis")
        
        for col_name, col_details in detailed_schema['column_details'].items():
            with st.expander(f"📊 {col_name} ({col_details['type']}) - {col_details['unique_count']:,} unique values"):
                
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.write(f"**Data Type:** {col_details['type']}")
                    st.write(f"**Unique Values:** {col_details['unique_count']:,}")
                    st.write(f"**Missing Values:** {col_details['null_count']} ({col_details['null_percentage']:.1f}%)")
                    st.write(f"**Categorical:** {'Yes' if col_details['is_categorical'] else 'No'}")
                
                with col_info2:
                    # Show sample values or statistics
                    if col_details['is_categorical']:
                        st.write("**Unique Values:**")
                        if len(col_details['sample_values']) <= 10:
                            for val in col_details['sample_values']:
                                st.write(f"• {val}")
                        else:
                            st.write("**Sample Values:**")
                            for val in col_details['sample_values'][:8]:
                                st.write(f"• {val}")
                            st.write(f"... and {col_details['unique_count'] - 8} more")
                    
                    elif col_details['stats']:
                        st.write("**Statistics:**")
                        st.write(f"Min: {col_details['stats']['min']:,.2f}")
                        st.write(f"Max: {col_details['stats']['max']:,.2f}")
                        st.write(f"Mean: {col_details['stats']['mean']:,.2f}")
                        st.write(f"Median: {col_details['stats']['median']:,.2f}")
    
    def show_intelligent_suggestions(self, detailed_schema):
        """Show intelligent query suggestions based on detailed analysis"""
        
        st.subheader("💡 Intelligent Query Suggestions")
        st.write("Based on your data analysis, here are some smart questions you can ask:")
        
        suggestions = []
        
        # Basic suggestions
        suggestions.extend(["Show all records", "Count total rows"])
        
        # Categorical column suggestions
        for col_name, col_details in detailed_schema['column_details'].items():
            if col_details['is_categorical'] and col_details['unique_count'] > 1:
                suggestions.append(f"Count records by {col_name}")
                if len(suggestions) >= 8:
                    break
        
        # Numeric column suggestions
        for col_name, col_details in detailed_schema['column_details'].items():
            if col_details['stats']:
                suggestions.extend([
                    f"What is the average {col_name}?",
                    f"Show records where {col_name} > {col_details['stats']['mean']:.0f}"
                ])
                if len(suggestions) >= 12:
                    break
        
        # Display suggestions in a nice format
        for i, suggestion in enumerate(suggestions[:10]):
            st.write(f"{i+1}. {suggestion}")
    
    def load_sample(self, file_path):
        """Load sample data with enhanced analysis"""
        if os.path.exists(file_path):
            with st.spinner("Loading sample data..."):
                result = self.csv_processor.load_csv(file_path)
            
            if result['success']:
                detailed_schema = self.get_detailed_schema_analysis()
                st.session_state.detailed_schema = detailed_schema
                
                st.success("✅ Sample data loaded!")
                self.show_comprehensive_overview(detailed_schema)
                
                if st.button("🚀 Start Querying Sample Data", key="start_sample"):
                    self.setup_enhanced_queries()
        else:
            st.error("Sample data not found. Run: python setup.py")
    
    def setup_enhanced_queries(self):
        """Setup enhanced query system"""
        with st.spinner("🛠️ Setting up high-accuracy query system..."):
            try:
                success = self.csv_processor.to_sqlite("data.db")
                
                if success:
                    st.session_state.data_ready = True
                    st.session_state.page = "query"
                    st.success("✅ High-accuracy query system ready!")
                    st.rerun()
                else:
                    st.error("❌ Setup failed")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    def query_page(self):
        """Enhanced query page with high accuracy"""
        
        # Initialize components
        if not self.schema_mapper:
            basic_schema = st.session_state.detailed_schema['basic_schema']
            self.schema_mapper = SchemaMapper(basic_schema)
            self.query_builder = QueryBuilder(basic_schema)
        
        st.title("💬 High-Accuracy Data Querying")
        
        # Data overview
        detailed_schema = st.session_state.detailed_schema
        basic_schema = detailed_schema['basic_schema']
        
        st.write(f"📊 **{basic_schema['total_rows']:,} rows** | **{len(basic_schema['column_names'])} columns**")
        
        # Show available columns with details
        with st.expander("📋 Available Columns & Values", expanded=False):
            for col_name, col_details in detailed_schema['column_details'].items():
                col_type = col_details['type']
                icon = "🔢" if col_type in ['INTEGER', 'REAL'] else "📝" if col_type == 'TEXT' else "📅"
                
                st.write(f"{icon} **{col_name}** ({col_type}) - {col_details['unique_count']:,} unique values")
                
                # Show sample values for categorical columns
                if col_details['is_categorical'] and len(col_details['sample_values']) <= 5:
                    st.write(f"   Values: {', '.join(map(str, col_details['sample_values']))}")
                elif col_details['stats']:
                    stats = col_details['stats']
                    st.write(f"   Range: {stats['min']:.1f} - {stats['max']:.1f} (avg: {stats['mean']:.1f})")
        
        # Back button
        if st.button("← Upload Different Data", key="back_btn"):
            st.session_state.page = "upload"
            st.session_state.detailed_schema = None
            st.rerun()
        
        st.markdown("---")
        
        # Smart examples based on detailed analysis
        st.write("**🎯 Smart Query Examples:**")
        examples = self.get_context_aware_examples(detailed_schema)
        
        cols = st.columns(2)
        for i, example in enumerate(examples[:6]):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}"):
                    self.process_enhanced_query(example)
        
        st.markdown("---")
        
        # Enhanced query input
        st.write("**🎯 Your Question:**")
        st.write("💡 Use the exact column names shown above for best results")
        
        question = st.text_input(
            "Ask anything about your data:", 
            placeholder="e.g., Count customers by gender",
            help="Use exact column names for better accuracy"
        )
        
        if st.button("🔍 Get High-Accuracy Answer", type="primary", key="get_answer") and question:
            self.process_enhanced_query(question)
    
    def get_context_aware_examples(self, detailed_schema):
        """Generate context-aware examples based on actual data"""
        
        examples = ["Show all records", "Count total rows"]
        
        # Add categorical examples
        categorical_cols = []
        for col_name, col_details in detailed_schema['column_details'].items():
            if col_details['is_categorical'] and 2 <= col_details['unique_count'] <= 20:
                categorical_cols.append(col_name)
        
        for col in categorical_cols[:2]:
            examples.append(f"Count records by {col}")
        
        # Add numeric examples
        numeric_cols = []
        for col_name, col_details in detailed_schema['column_details'].items():
            if col_details['stats']:
                numeric_cols.append((col_name, col_details['stats']))
        
        for col_name, stats in numeric_cols[:2]:
            examples.extend([
                f"What is the average {col_name}?",
                f"Show records where {col_name} > {stats['mean']:.0f}"
            ])
        
        return examples[:8]
    
    def process_enhanced_query(self, question):
        """Enhanced query processing with better accuracy"""
        
        st.markdown("---")
        st.write(f"**Question:** _{question}_")
        
        try:
            with st.spinner("🧠 Processing with high accuracy..."):
                # Enhanced processing
                processed = self.text_processor.process_query(question)
                mapped_columns = self.enhanced_column_mapping(question, processed)
                conditions = self.enhanced_condition_extraction(question, processed, mapped_columns)
                query_result = self.query_builder.build_query(processed, mapped_columns, conditions)
            
            # Show detailed understanding
            with st.expander("🧠 Detailed Query Analysis", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Intent:** {processed['intent']}")
                    st.write(f"**Keywords:** {processed['keywords']}")
                with col2:
                    st.write(f"**Mapped columns:** {mapped_columns if mapped_columns else 'All columns'}")
                    if processed['numbers']:
                        st.write(f"**Numbers found:** {[n['value'] for n in processed['numbers']]}")
                with col3:
                    if conditions:
                        st.write(f"**Conditions:** {len(conditions)} filter(s)")
                        for cond in conditions:
                            st.write(f"  {cond['column']} {cond['operator']} {cond['value']}")
            
            # Show SQL with confidence score
            st.write("**🔧 Generated SQL Query:**")
            st.code(query_result['sql'], language="sql")
            
            # Calculate and show confidence
            confidence = self.calculate_query_confidence(question, mapped_columns, conditions, processed)
            if confidence >= 0.8:
                st.success(f"🎯 High Confidence ({confidence*100:.0f}%): {query_result['explanation']}")
            elif confidence >= 0.6:
                st.warning(f"⚠️ Medium Confidence ({confidence*100:.0f}%): {query_result['explanation']}")
            else:
                st.error(f"❓ Low Confidence ({confidence*100:.0f}%): Query may not be accurate")
            
            # Execute and show results
            st.write("**📊 Query Results:**")
            self.execute_and_display_results(query_result['sql'])
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Try using exact column names from the 'Available Columns' section above")
    
    def enhanced_column_mapping(self, question, processed):
        """Enhanced column mapping with context awareness"""
        detailed_schema = st.session_state.detailed_schema
        column_details = detailed_schema['column_details']
        
        question_lower = question.lower()
        mapped = []
        
        # Direct exact matches (highest priority)
        for col_name in column_details.keys():
            if col_name.lower() in question_lower:
                mapped.append(col_name)
        
        # Keyword matching with fuzzy logic
        from fuzzywuzzy import fuzz
        for keyword in processed['keywords']:
            for col_name in column_details.keys():
                similarity = fuzz.ratio(keyword.lower(), col_name.lower())
                if similarity > 85:  # High similarity threshold
                    if col_name not in mapped:
                        mapped.append(col_name)
        
        return mapped
    
    def enhanced_condition_extraction(self, question, processed, mapped_columns):
        """Enhanced condition extraction with better accuracy"""
        conditions = []
        comparisons = processed.get('comparison_operators', [])
        
        if comparisons and mapped_columns:
            detailed_schema = st.session_state.detailed_schema
            
            for comp in comparisons:
                # Use the most appropriate column for the condition
                column = mapped_columns[0]
                col_details = detailed_schema['column_details'][column]
                
                conditions.append({
                    'column': column,
                    'operator': comp['sql_operator'],
                    'value': comp['value'],
                    'type': col_details['type']
                })
        
        return conditions
    
    def calculate_query_confidence(self, question, mapped_columns, conditions, processed):
        """Calculate confidence score for the generated query"""
        confidence = 0.0
        
        # Intent recognition confidence
        if processed['intent'] != 'SELECT':
            confidence += 0.3
        else:
            confidence += 0.2
        
        # Column mapping confidence
        if mapped_columns:
            confidence += 0.4
        else:
            confidence += 0.1
        
        # Condition extraction confidence
        if processed.get('comparison_operators') and conditions:
            confidence += 0.3
        elif not processed.get('comparison_operators'):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def execute_and_display_results(self, sql):
        """Execute query and display results with enhanced formatting"""
        try:
            conn = sqlite3.connect("data.db")
            result_df = pd.read_sql_query(sql, conn)
            conn.close()
            
            if len(result_df) > 0:
                # Handle single value results
                if len(result_df) == 1 and len(result_df.columns) == 1:
                    answer = result_df.iloc[0, 0]
                    if isinstance(answer, (int, float)):
                        st.markdown(f"### 🎯 Answer: **{answer:,.2f}**")
                    else:
                        st.markdown(f"### 🎯 Answer: **{answer}**")
                
                # Show data table with better formatting
                st.dataframe(result_df, use_container_width=True)
                
                # Show result summary
                if len(result_df) > 1:
                    st.success(f"✅ Found **{len(result_df):,} results**")
                    
                    # Show additional insights for GROUP BY results
                    if len(result_df.columns) == 2 and 'count' in result_df.columns:
                        total = result_df['count'].sum()
                        st.info(f"📊 Total count across all groups: **{total:,}**")
                
            else:
                st.warning("📭 No results found for your query")
                st.info("💡 Try adjusting your question or check if the values exist in your data")
                
        except Exception as e:
            st.error(f"❌ Query execution error: {str(e)}")


def main():
    app = HighAccuracyNL2SQLApp()
    app.run()

if __name__ == "__main__":
    main()