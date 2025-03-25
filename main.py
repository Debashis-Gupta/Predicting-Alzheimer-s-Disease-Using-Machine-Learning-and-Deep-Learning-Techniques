import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from eda import *
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
from neural import train_neural_network, predict_with_neural_network
from resnet import train_resnet_network, predict_with_resnet_network
from traditional import train_traditional_ml
import os
import torch
import joblib
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Define paths relative to the script's directory
CURRENT_DIR = os.getcwd()
EDA_FIG_PATH = os.path.join(CURRENT_DIR, "EDA_Figures")
CSV_PATH = os.path.join(CURRENT_DIR, "alzheimers_disease_data.csv")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
def read_csv(csv_path):
    """Read CSV file with error handling using absolute path."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}. Please check the path and ensure the file exists.")
    return pd.read_csv(csv_path)

def selecteKBestFeatures(X, y, topK=10):
    selector = SelectKBest(f_classif, k=topK)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    print(f"Top {topK} selected features:", selected_features)
    return X_selected, selected_features

def PCA_Visualization(data):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data.drop(columns=['Diagnosis']))
    
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Diagnosis'] = data['Diagnosis'].values
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Diagnosis', data=pca_df)
    plt.title('PCA: 2 Component Visualization')
    plt.savefig(os.path.join(EDA_FIG_PATH, "PCA_visualization.jpeg"), dpi=300)
    plt.close()

def tsne_visualization(data):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(data.drop(columns=['Diagnosis']))
    
    tsne_df = pd.DataFrame(data=X_tsne, columns=['t-SNE1', 't-SNE2'])
    tsne_df['Diagnosis'] = data['Diagnosis'].values
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='Diagnosis', data=tsne_df)
    plt.title('t-SNE: 2 Component Visualization')
    plt.savefig(os.path.join(EDA_FIG_PATH, "TSNE_visualization.jpeg"), dpi=300)
    plt.close()

def evaluate_model(y_true, y_pred, y_prob, model_name):
    print(f"\n{model_name} Evaluation Metrics:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_prob[:, 1]))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

def main():
    # Ensure EDA_FIG_PATH directory exists
    os.makedirs(EDA_FIG_PATH, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Read the CSV file
    data = read_csv(CSV_PATH)

    show_null_values(data)
    print("-"*100, flush=True)
    any_duplicate(data)
    print("-"*100, flush=True)
    get_columns(data, show=True)
    print("-"*100, flush=True)
    show_column_type(data)
    print("-"*100, flush=True)
    print(data.sample(10))
    print("-"*100, flush=True)
    dropped_column_list = ['DoctorInCharge']
    drop_columns(data, columns=dropped_column_list)
    print("-"*100, flush=True)

    numerical_columns = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 
                         'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 
                         'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE']
    detecting_outlier(data, numerical_columns, outdir_path=EDA_FIG_PATH, filename="OutlierDetection")    
    target_col = 'Age'
    visualize_against_disease(data, target_col, EDA_FIG_PATH, filename=f"Distribution_{target_col}_Against_Disease", histogram=True)
    target_col = 'Gender'
    visualize_against_disease(data, target_col, EDA_FIG_PATH, filename=f"Distribution_{target_col}_Against_Disease", Bar=True)
    target_col = 'FamilyHistoryAlzheimers'
    visualize_against_disease(data, target_col, EDA_FIG_PATH, filename=f"Distribution_{target_col}_Against_Disease", Pie=True)
    visualize_correlation(data, numerical_columns, EDA_FIG_PATH, "Correlation_matrix")
    show_cholesterol_visualiation(data, EDA_FIG_PATH, 'Relationship Between Age and Cholesterol')

    plot_target_distribution(
        data=data, 
        target_column='Diagnosis', 
        outdir_path=EDA_FIG_PATH, 
        filename='Target_Distribution',
        title='Distribution of Alzheimer\'s Disease in Dataset'
    )

    # Prepare data for ML models
    X = data.drop(columns=['PatientID', 'Diagnosis','DoctorInCharge'])
    y = data['Diagnosis']
    
    # Preprocessing and Feature Selection
    pt = PowerTransformer()
    X[numerical_columns] = pt.fit_transform(X[numerical_columns])
    X_selected, selected_features = selecteKBestFeatures(X, y, topK=10)
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define output directories
    traditional_outdir = os.path.join(CURRENT_DIR, "traditional_output")
    nn_outdir = os.path.join(CURRENT_DIR, "nn_output")
    resnet_outdir = os.path.join(CURRENT_DIR, "resnet_output")
    os.makedirs(traditional_outdir, exist_ok=True)
    os.makedirs(nn_outdir, exist_ok=True)
    os.makedirs(resnet_outdir, exist_ok=True)

    print("\n" + "="*50)
    print("Training Traditional ML Models with Grid Search")
    print("="*50)
    rf_model,xgb_model = train_traditional_ml(X_train_scaled, X_test_scaled, y_train, y_test, outdir_path=traditional_outdir)
    rf_model_path = os.path.join(MODELS_DIR, "random_forest.pth")
    xgb_model_path = os.path.join(MODELS_DIR, "xgboost.pth")
    joblib.dump(rf_model, rf_model_path)
    joblib.dump(xgb_model, xgb_model_path)
    print(f"Random Forest model saved to: {rf_model_path}")
    print(f"XGBoost model saved to: {xgb_model_path}")
    # Neural Network Training
    print("\n" + "="*50)
    print("Training Neural Network with Selected Features")
    print("="*50)
    print(f"Neural Network - X_selected shape: {X_selected.shape}, y shape: {y.shape}")
    nn_model, nn_fold_histories, nn_fold_metrics = train_neural_network(
        X=X_selected,  # Full dataset for k-fold cross-validation
        y=y.values, 
        outdir_path=nn_outdir,
        batch_size=32,
        epochs=100,
        learning_rate=0.01,
        k_folds=5
    )
    nn_preds, nn_probs = predict_with_neural_network(nn_model, X_test_scaled)
    print(f"Neural Network - X_test shape: {X_test_scaled.shape}, y_test shape: {y_test.shape}")
    evaluate_model(y_test, nn_preds, nn_probs, "Neural Network")
    nn_model_path = os.path.join(MODELS_DIR, "neural_network.pth")
    torch.save(nn_model.state_dict(), nn_model_path)
    print(f"Neural Network model saved to: {nn_model_path}")

    # ResNet-50 Training
    print("\n" + "="*50)
    print("Training ResNet-50 with Selected Features")
    print("="*50)
    print(f"ResNet-50 - X_selected shape: {X_selected.shape}, y shape: {y.shape}")
    resnet_model, resnet_fold_histories, resnet_fold_metrics = train_resnet_network(
        X=X_selected,  # Full dataset for k-fold cross-validation
        y=y.values, 
        outdir_path=resnet_outdir,
        batch_size=32,
        epochs=100,
        learning_rate=0.01,
        k_folds=5
    )
    resnet_preds, resnet_probs = predict_with_resnet_network(resnet_model, X_test_scaled)
    print(f"ResNet-50 - X_test shape: {X_test_scaled.shape}, y_test shape: {y_test.shape}")
    evaluate_model(y_test, resnet_preds, resnet_probs, "ResNet-50")
    # Save ResNet-50 model
    resnet_model_path = os.path.join(MODELS_DIR, "resnet50.pth")
    torch.save(resnet_model.state_dict(), resnet_model_path)
    print(f"ResNet-50 model saved to: {resnet_model_path}")

if __name__ == "__main__":
    main()