import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc

def shap_analysis(model, X_train, X_test):
    # Check if the model is trained
    if not hasattr(model, "estimators_"):
        print("Model is not trained. Please train the model before SHAP analysis.")
        return
    
    # Use TreeExplainer for tree-based models like RandomForest
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot for the positive class (e.g., class 1 for binary classification)
    shap.summary_plot(shap_values[1], X_test)
    
    # Print SHAP values for the first test instance as an example
    print("SHAP values for the first test instance (Class 1):")
    print(shap_values[1][0])


def plot_feature_importance(model, X_train):
    """
    Generates a feature importance plot using SHAP values.
    """
    import shap
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Ensure model is trained
    if not hasattr(model, "fit"):
        print("ERROR: Model is not trained. Train the model before plotting feature importance.")
        return
    
    # Use TreeExplainer for tree-based models (RandomForest, XGBoost)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Check if shap_values is a list (for classification models)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take values for class 1

    # Compute mean absolute SHAP values per feature
    shap_importance = np.abs(shap_values).mean(axis=0)

    # Ensure shap_importance is 1D
    shap_importance = np.ravel(shap_importance)  # Flatten it

    # Get feature names
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns
    else:
        feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

    # Ensure lengths match
    if len(feature_names) != len(shap_importance):
        print("WARNING: Mismatch in feature names and SHAP values.")
        min_length = min(len(feature_names), len(shap_importance))
        feature_names = feature_names[:min_length]
        shap_importance = shap_importance[:min_length]

    # Print lengths to check
    print(f"Feature names count: {len(feature_names)}")
    print(f"SHAP importance values count: {len(shap_importance)}")

    # Create DataFrame for visualization
    importance_df = pd.DataFrame({'Feature': feature_names, 'SHAP Importance': shap_importance})
    importance_df = importance_df.sort_values(by="SHAP Importance", ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x="SHAP Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title("Feature Importance (SHAP Values)")
    plt.xlabel("Mean |SHAP Value|")
    plt.ylabel("Feature")
    plt.show()



# def plot_log_ensemble_auc(y_test, ensemble_probs):
#     fpr, tpr, _ = roc_curve(y_test, ensemble_probs)
#     roc_auc = auc(fpr, tpr)

#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, color='blue', lw=2, label=f'Logarithmic Ensemble (AUC = {roc_auc:.2f})')
    
#     # Plot random guessing line
#     plt.plot([0, 1], [0, 1], linestyle='dashed', color='gray', label="Random Guessing (AUC = 0.50)")

#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("ROC Curve for Logarithmic Ensemble Model")
#     plt.legend(loc="lower right")
#     plt.grid()
#     plt.show()



def plot_roc_curves(y_test, probabilities):
    """
    Plots ROC curves for all models and the ensemble.

    Args:
        y_test (array-like): True binary labels.
        probabilities (dict): Dictionary of predicted probabilities for each model.
    """
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')  # Random guess line

    for name, prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    # Plot ensemble ROC
    ensemble_probs = logarithmic_ensemble(probabilities, {
        'Logistic Regression': 0.2,
        'Random Forest': 0.3,
        'XGBoost': 0.3,
        'ANN': 0.2
    })
    fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, ensemble_probs)
    roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble)
    plt.plot(fpr_ensemble, tpr_ensemble, label=f'Logarithmic Ensemble (AUC = {roc_auc_ensemble:.2f})', linewidth=2)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.show()