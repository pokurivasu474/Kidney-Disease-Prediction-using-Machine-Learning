                     Kidney Disease Prediction using Machine Learning
Description:
This project predicts whether a patient has Chronic Kidney Disease (CKD) using machine learning models.
It uses clinical and medical features from a dataset and evaluates multiple models to find the best performing one.

Objectives:
Build classification models for disease prediction
Compare models using evaluation metrics
Perform hyperparameter tuning for better performance
Select the best model based on ROC-AUC score

Technologies:
Python
Pandas
NumPy
Scikit-learn

Workflow:
Load dataset from CSV file
Perform data preprocessing
Handle missing values
Convert data types
Feature selection
Train multiple machine learning models
Hyperparameter tuning using GridSearchCV
Evaluate models using performance metrics
Select the best model

Models:
Logistic Regression
Random Forest
Support Vector Machine (SVM)

Evaluation Metrics:
Accuracy
Precision
Recall
F1 Score
ROC-AUC Score

How to Run the Project
Step 1: Clone the repository
git clone 
cd ML_PROJECT
Step 2: Install dependencies
pip install -r requirements.txt
Step 3: Run the project
python src/main.py

Results:
The models were evaluated using multiple metrics.
The best model was selected based on the highest ROC-AUC score
