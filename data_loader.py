import pandas as pd
def load_data(filepath):
    # Specify strings to be interpreted as missing values
    missing_values = ['?', '\t?']
    df = pd.read_csv(filepath, na_values=missing_values)
    
    # Remove extra whitespace in column names
    df.columns = df.columns.str.strip()
    
    # Map target variable: "ckd" -> 1, "notckd" -> 0
    df['classification'] = df['classification'].str.strip().map({'ckd': 1, 'notckd': 0})
    
    # Convert numeric columns to numeric types
    numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
                    'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_data(df):
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("Class distribution:")
    print(df['classification'].value_counts(normalize=True))