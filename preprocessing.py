import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(df):
    # Drop identifier and separate target
    X = df.drop([ 'classification'], axis=1)
    y = df['classification']
    
    # Define features based on dataset columns:
    # Numeric features
    numeric_features = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
                        'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    
    # Categorical features
    categorical_features = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 
                            'appet', 'pe', 'ane']
    

    plt.figure(figsize=(12, 8))
    corr_matrix = df[numeric_features + ['classification']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    
    # Create preprocessing pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ])
    return preprocessor, X, y