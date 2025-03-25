import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
import xgboost as xgb
import matplotlib.pyplot as plt

def train_traditional_ml(X_train, X_test, y_train, y_test, outdir_path, topK=10):
    print(f"Input X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Input X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # --- RandomForestClassifier ---
    print(f"####### TRAINING RANDOM FOREST WITH PROVIDED FEATURES ###########")
    
    # Define parameter grid for RandomForestClassifier
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
    rf_grid_search.fit(X_train, y_train)
    
    print("\nBest Parameters for Random Forest:", rf_grid_search.best_params_)
    best_rf = rf_grid_search.best_estimator_
    rf_y_pred = best_rf.predict(X_test)
    rf_y_prob = best_rf.predict_proba(X_test)
    
    # Compute metrics for Random Forest
    rf_accuracy = accuracy_score(y_test, rf_y_pred)
    rf_f1 = f1_score(y_test, rf_y_pred)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_prob[:, 1])
    rf_roc_auc = auc(rf_fpr, rf_tpr)
    
    # Print metrics for Random Forest
    print("Random Forest - Classification Report:\n", classification_report(y_test, rf_y_pred))
    print("Random Forest - Confusion Matrix:\n", confusion_matrix(y_test, rf_y_pred))
    print(f"Random Forest - Accuracy Score: {rf_accuracy:.4f}")
    print(f"Random Forest - F1 Score: {rf_f1:.4f}")
    print(f"Random Forest - ROC-AUC Score: {rf_roc_auc:.4f}")
    
    # K-fold cross-validation for Random Forest
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    rf_cv_scores = cross_val_score(best_rf, X_train, y_train, cv=skf, scoring='accuracy')
    
    print(f"\nRandom Forest - K-Fold Cross Validation Scores (k={n_folds}):", rf_cv_scores)
    print(f"Random Forest - Mean Accuracy: {rf_cv_scores.mean():.4f}")
    print(f"Random Forest - Standard Deviation: {rf_cv_scores.std():.4f}")
    
    # Plot ROC curve for Random Forest
    plt.figure()
    plt.plot(rf_fpr, rf_tpr, label=f'ROC curve (AUC = {rf_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest')
    plt.legend(loc="lower right")
    plt.savefig(f"{outdir_path}/roc_curve_rf.jpeg", dpi=300)
    plt.close()
    
    # --- XGBoostClassifier ---
    print(f"\n####### TRAINING XGBOOST WITH PROVIDED FEATURES ###########")
    
    # Define parameter grid for XGBoostClassifier
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    }
    
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_grid_search = GridSearchCV(estimator=xgb_clf, param_grid=xgb_param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
    xgb_grid_search.fit(X_train, y_train)
    
    print("\nBest Parameters for XGBoost:", xgb_grid_search.best_params_)
    best_xgb = xgb_grid_search.best_estimator_
    xgb_y_pred = best_xgb.predict(X_test)
    xgb_y_prob = best_xgb.predict_proba(X_test)
    
    # Compute metrics for XGBoost
    xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
    xgb_f1 = f1_score(y_test, xgb_y_pred)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_y_prob[:, 1])
    xgb_roc_auc = auc(xgb_fpr, xgb_tpr)
    
    # Print metrics for XGBoost
    print("XGBoost - Classification Report:\n", classification_report(y_test, xgb_y_pred))
    print("XGBoost - Confusion Matrix:\n", confusion_matrix(y_test, xgb_y_pred))
    print(f"XGBoost - Accuracy Score: {xgb_accuracy:.4f}")
    print(f"XGBoost - F1 Score: {xgb_f1:.4f}")
    print(f"XGBoost - ROC-AUC Score: {xgb_roc_auc:.4f}")
    
    # K-fold cross-validation for XGBoost
    xgb_cv_scores = cross_val_score(best_xgb, X_train, y_train, cv=skf, scoring='accuracy')
    
    print(f"\nXGBoost - K-Fold Cross Validation Scores (k={n_folds}):", xgb_cv_scores)
    print(f"XGBoost - Mean Accuracy: {xgb_cv_scores.mean():.4f}")
    print(f"XGBoost - Standard Deviation: {xgb_cv_scores.std():.4f}")
    
    # Plot ROC curve for XGBoost
    plt.figure()
    plt.plot(xgb_fpr, xgb_tpr, label=f'ROC curve (AUC = {xgb_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - XGBoost')
    plt.legend(loc="lower right")
    plt.savefig(f"{outdir_path}/roc_curve_xgb.jpeg", dpi=300)
    plt.close()
    
    return best_rf, best_xgb

# if __name__ == "__main__":
#     # Example usage (for testing traditional.py independently)
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     from sklearn.preprocessing import StandardScaler
#     data = pd.read_csv("Assignement1/Alzhemier_Code/alzheimers_disease_data.csv")
#     data = data.drop(columns=['DoctorInCharge'])
#     X = data.drop(columns=['PatientID', 'Diagnosis'])
#     y = data['Diagnosis']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#     rf_model, xgb_model = train_traditional_ml(X_train, X_test, y_train, y_test, outdir_path="traditional_output")