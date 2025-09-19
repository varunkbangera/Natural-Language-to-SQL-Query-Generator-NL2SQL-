# setup.py

import pandas as pd
import os

def create_sample_data():
    """Create sample CSV files"""
    
    # Employee data
    employees = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000, 60000, 55000, 65000, 58000],
        'department': ['Engineering', 'Marketing', 'Sales', 'Engineering', 'Marketing']
    })
    
    # Sales data
    sales = pd.DataFrame({
        'order_id': [101, 102, 103, 104, 105],
        'customer': ['Apple', 'Google', 'Microsoft', 'Amazon', 'Netflix'],
        'product': ['Laptop', 'Phone', 'Tablet', 'Laptop', 'Phone'],
        'revenue': [1200, 800, 600, 1200, 800]
    })
    
    # Student data
    students = pd.DataFrame({
        'student_id': [1001, 1002, 1003, 1004, 1005],
        'name': ['John', 'Jane', 'Mike', 'Sarah', 'Tom'],
        'subject': ['Math', 'Science', 'Math', 'Science', 'Math'],
        'score': [85, 92, 78, 88, 95]
    })
    
    # Save files
    os.makedirs('data/sample_datasets', exist_ok=True)
    
    employees.to_csv('data/sample_datasets/employees.csv', index=False)
    sales.to_csv('data/sample_datasets/sales.csv', index=False)
    students.to_csv('data/sample_datasets/students.csv', index=False)
    
    print("✅ Sample data created:")
    print("  - data/sample_datasets/employees.csv")
    print("  - data/sample_datasets/sales.csv")
    print("  - data/sample_datasets/students.csv")

def main():
    print("🚀 Setting up NL2SQL project...")
    create_sample_data()
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("1. Run: pip install -r requirements.txt")
    print("2. Run: streamlit run src/ui/streamlit_app.py")

if __name__ == "__main__":
    main()