import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from data_loader import load_data, validate_data
from preprocessing import preprocess_data
from feature_selection import feature_selection
from train import train_evaluate_models, logarithmic_ensemble
from evaluation import plot_feature_importance, plot_roc_curves


if __name__ == "__main__":
    # Load and validate the dataset
    df = load_data('kidney_records_5000.csv')
    validate_data(df)
    
    # Preprocessing
    preprocessor, X, y = preprocess_data(df)
    X_processed = preprocessor.fit_transform(X)
    
    # Apply feature selection
    print("Applying feature selection...")
    selector, X_selected = feature_selection(pd.DataFrame(X_processed), y, k=15)


    # selector, X_selected = feature_selection(X_processed, y, preprocessor, k=15)


    # Save the feature selector
    joblib.dump(selector, "feature_selector.pkl")

    # Debugging: Ensure the file is saved
    import os
    if os.path.exists("feature_selector.pkl"):
        print("Feature selector saved successfully.")
    else:
        print("ERROR: Feature selector not saved!")
    
    # Address class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_selected, y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
    
    
    
    # Train individual models with hyperparameter tuning
    results, probabilities, models = train_evaluate_models(X_train, X_test, y_train, y_test)
    

# Extract the trained Random Forest model
    rf_trained_model = models['Random Forest']['model'].fit(X_train, y_train)

    # Now, pass the trained model to SHAP
    plot_feature_importance(models['Random Forest']['model'], pd.DataFrame(X_train, columns=selector.get_feature_names_out()))





    # Logarithmic ensembling of model predictions
    ensemble_weights = {
        'Logistic Regression': 0.2,
        'Random Forest': 0.3,
        'XGBoost': 0.3,
        'ANN': 0.2
    }
    ensemble_probs = logarithmic_ensemble(probabilities, ensemble_weights)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    ensemble_metrics = {
        'Accuracy': accuracy_score(y_test, ensemble_preds),
        'Precision': precision_score(y_test, ensemble_preds),
        'Recall': recall_score(y_test, ensemble_preds),
        'F1': f1_score(y_test, ensemble_preds),
        'AUC-ROC': roc_auc_score(y_test, ensemble_probs)
    }
    results['Logarithmic Ensemble'] = ensemble_metrics
    
    

    # # Compute ROC curve and AUC for the logarithmic ensemble model
    # fpr, tpr, _ = roc_curve(y_test, ensemble_probs)
    # roc_auc = auc(fpr, tpr)

    # # Plot the ROC curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='blue', lw=2, label=f'Logarithmic Ensemble (AUC = {roc_auc:.3f})')
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # Diagonal reference line
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve - Logarithmic Ensemble Model')
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.show()

    # Build a stacking ensemble to further boost performance while controlling complexity
    # stacking_metrics, stacking_model = stacking_ensemble(X_train, X_test, y_train, y_test, models)
    # results['Stacking Ensemble'] = stacking_metrics
    # Compute probability predictions for each model


    
    plot_roc_curves(y_test, probabilities)



    # 2. Accuracy Comparison Bar Plot
    # Extract accuracy scores from results dictionary
    model_names = list(results.keys())
    accuracy_scores = [results[model]['Accuracy'] for model in model_names]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=model_names, y=accuracy_scores, palette='viridis')
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.show()

   
    
    # Display performance metrics
    results_df = pd.DataFrame(results).T.drop(columns=['Best Params'], errors='ignore')
    print("Model Performance Comparison:")
    print(results_df)
    
    # (Optional) Run SHAP analysis on one of the models – for example, the Random Forest
    # shap_analysis(models['Random Forest']['model'], X_train, X_test)
    
    # PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_selected)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title('PCA Visualization of CKD Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    




    import joblib

    joblib.dump(preprocessor,"preprocessor.pkl")
    print("Preprocessor saved successfully.")

   # Extract and save only the trained model instances
    trained_models = {}

    for name, config in models.items():
        print(f"Training {name}...")
        config["model"].fit(X_train, y_train)  # Ensure models are trained
        trained_models[name] = config["model"]  # Store only trained models

    joblib.dump(trained_models, "trained_models.pkl")
    print("Trained models saved successfully!")

    import os
    print("Saved model size:", os.path.getsize("trained_models.pkl"), "bytes")

    # Train and save stacking ensemble
    # stacking_model = StackingClassifier(
    #     estimators=[(name, config["model"]) for name, config in models.items()],
    #     final_estimator=LogisticRegression(),
    #     cv=5,
    #     n_jobs=-1
    # )

    # print("Training Stacking Ensemble...")
    # stacking_model.fit(X_train, y_train)
    # print(" Stacking Ensemble trained successfully.")

    # # Save stacking model and verify
    # joblib.dump(stacking_model, "stacking_ensemble.pkl")

    # import os
    # if os.path.exists("stacking_ensemble.pkl"):
    #     print(" Stacking ensemble model saved successfully.")
    # else:
    #     print(" ERROR: Stacking ensemble model was not saved!")


    # Ensure the best model is actually trained before saving
    # if 'models' in locals() and 'Random Forest' in models:
    #     best_model = models['Random Forest']['model']

    #     print("Training the model before saving...")
    #     best_model.fit(X_train, y_train)  # Manually train it

    #     print("Saving model...")
    #     joblib.dump(best_model, "kidney_disease_model.pkl")

    #     # Verify the saved file size
    #     import os
    #     print("Saved model size:", os.path.getsize("kidney_disease_model.pkl"), "bytes")
    # else:
    #     print("ERROR: Model training failed, no model found!")