import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
def train_evaluate_models(X_train, X_test, y_train, y_test):
    # Updated hyperparameter grids to reduce model complexity and overfitting
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            'params': {
                'C': [0.01, 0.1],  # Lower C means stronger regularization
                'solver': ['liblinear']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [5, 7, 10],  # Limit tree depth
                'min_samples_split': [2, 5]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                                    eval_metric='logloss', random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [3, 4, 5],  # Lower depth for better generalization
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
        },
        'ANN': {
            'model': MLPClassifier(early_stopping=True, random_state=42, max_iter=500),
            'params': {
                'hidden_layer_sizes': [(32,), (32, 16)],  # Simpler network architectures
                'alpha': [0.001, 0.01]  # Increased regularization
            }
        }
    }
        # Check if models are trained
    if 'models' in locals():
        for name, config in models.items():
            print(f"Model: {name} - Trained: {hasattr(config['model'], 'fit')}")
    else:
        print("ERROR: models dictionary is empty!")
    results = {}
    probabilities = {}

    
    for name, config in models.items():
        print(f"Optimizing {name}...")
        gs = GridSearchCV(config['model'], config['params'],
                          cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_

        
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        probabilities[name] = y_proba
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_proba),
            'Best Params': gs.best_params_  
        }
    return results, probabilities, models

# def shap_analysis(model, X_train, X_test):
#     # Initialize SHAP explainer
#     explainer = shap.Explainer(model, X_train)
    
#     # Compute SHAP values
#     shap_values = explainer(X_test)
    
#     # Summary plot
#     shap.summary_plot(shap_values, X_test)
    
#     # Print SHAP values for the first instance as an example
#     print("SHAP values for the first test instance:")
#     print(shap_values[0].values)



# def stacking_ensemble(X_train, X_test, y_train, y_test, models):
#     # Build a stacking ensemble using the tuned base estimators
#     estimators = []
#     for name, config in models.items():
#         estimator = config['model']
#         estimators.append((name, estimator))
    
#     stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5, n_jobs=-1)
#     stack.fit(X_train, y_train)
#     y_pred = stack.predict(X_test)
#     y_proba = stack.predict_proba(X_test)[:, 1]
#     metrics = {
#         'Accuracy': accuracy_score(y_test, y_pred),
#         'Precision': precision_score(y_test, y_pred),
#         'Recall': recall_score(y_test, y_pred),
#         'F1': f1_score(y_test, y_pred),
#         'AUC-ROC': roc_auc_score(y_test, y_proba)
#     }
#     return metrics, stack

def logarithmic_ensemble(prob_dict, weights):
    total_weight = sum(weights.values())
    weighted_log_sum = 0.0
    epsilon = 1e-15  # Avoid log(0)
    for model_name, prob in prob_dict.items():
        if model_name in weights:
            prob = np.clip(prob, epsilon, 1 - epsilon)
            weighted_log_sum += weights[model_name] * np.log(prob)
    return np.exp(weighted_log_sum / total_weight)