import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from eda import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from eda import plot_target_distribution
from neural import train_neural_network
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


EDA_FIG_PATH = "/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Assignement1/Alzhemier_Code/EDA_Figures/"
CSV_PATH = "/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Assignement1/Alzhemier_Code/alzheimers_disease_data.csv"

def read_csv(CSV_PATH):
    return pd.read_csv(CSV_PATH)


def selecteKBestFeatures(X, y, topK=10):
    # Feature Selection using SelectKBest
    selector = SelectKBest(f_classif, k=topK)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    print(f"Top {topK} selected features:", selected_features)
    return selected_features

def PCA_Visualization(data):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)
    
    # Create PCA plot
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Diagnosis'] = data['Diagnosis'].values()
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Diagnosis', data=pca_df)
    plt.title('PCA: 2 Component Visualization')
    plt.savefig(f"{EDA_FIG_PATH}/PCA_visualization.jpeg", dpi=300)
    plt.close()

def tsne_visualization(data):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(data)
    
    # Create t-SNE plot
    tsne_df = pd.DataFrame(data=X_tsne, columns=['t-SNE1', 't-SNE2'])
    tsne_df['Diagnosis'] = data['Diagnosis'].values()
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='Diagnosis', data=tsne_df)
    plt.title('t-SNE: 2 Component Visualization')
    plt.savefig(f"{EDA_FIG_PATH}/TSNE_visualization.jpeg", dpi=300)
    plt.close()


def main():
    data = read_csv(CSV_PATH=CSV_PATH)

    show_null_values(data)
    print("-"*100,flush=True)
    any_duplicate(data)
    print("-"*100,flush=True)
    get_columns(data,show=True)
    print("-"*100,flush=True)
    show_column_type(data)
    print("-"*100,flush=True)
    print(data.sample(10))
    print("-"*100,flush=True)
    dropped_column_list=['DoctorInCharge']
    drop_columns(data,columns=dropped_column_list)
    print("-"*100,flush=True)

    # Columns selected for visualization
    numerical_columns = ['Age', 'BMI', 
        'AlcoholConsumption', 'PhysicalActivity', 
        'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 
        'CholesterolLDL', 'CholesterolHDL', 
        'CholesterolTriglycerides', 'MMSE']
    detecting_outlier(data,numerical_columns,outdir_path=EDA_FIG_PATH,filename="OutlierDetection")    
    target_col='Age'
    visualize_against_disease(data,target_col,EDA_FIG_PATH,filename=f"Distribution_{target_col}_Against_Disease",histogram=True)
    target_col='Gender'
    visualize_against_disease(data,target_col,EDA_FIG_PATH,filename=f"Distribution_{target_col}_Against_Disease",Bar=True)
    target_col='FamilyHistoryAlzheimers'
    visualize_against_disease(data,target_col,EDA_FIG_PATH,filename=f"Distribution_{target_col}_Against_Disease",Pie=True)
    visualize_correlation(data,numerical_columns,EDA_FIG_PATH,"Correlation_matrix")
    show_cholesterol_visualiation(data,EDA_FIG_PATH,'Relationship Between Age and Cholesterol')

    # টার্গেট ক্লাসের ডিস্ট্রিবিউশন দেখানো
    plot_target_distribution(
        data=data, 
        target_column='Diagnosis', 
        outdir_path=EDA_FIG_PATH, 
        filename='Target_Distribution',
        title='Distribution of Alzheimer\'s Disease in Dataset'
    )

    #### WORK WITH ML MODEL ######
    print(f"####### SELECTING ALL FEATURES ###########")
    pt = PowerTransformer()
    data[numerical_columns] = pt.fit_transform(data[numerical_columns])
    # Preprocessing
    X_all = data.drop(columns=['PatientID', 'Diagnosis', 'DoctorInCharge'])
    y = data['Diagnosis']
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Train a model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy Score:\n", accuracy_score(y_test, y_pred))

    print(f"####### SELECTING top 10 FEATURES ###########")
    print("#"*30)
    selected_features = selecteKBestFeatures(X_all, y,topK=10)
    print(f'Selected Features : {selected_features}')

    data[selected_features] = pt.fit_transform(data[selected_features])
    # Preprocessing
    X_select = data[selected_features]
    y = data['Diagnosis']
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_select, y, test_size=0.2, random_state=42)
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Train a model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy Score:\n", accuracy_score(y_test, y_pred))

    # K-fold cross validation
    n_folds = 5
    # StratifiedKFold ব্যবহার করা হচ্ছে কারণ এটি টার্গেট ক্লাসের ভারসাম্য বজায় রাখে
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # মডেল তৈরি
    model = RandomForestClassifier(random_state=42)
    
    # ক্রস ভ্যালিডেশন স্কোর গণনা
    cv_scores = cross_val_score(model, X_select, y, cv=skf, scoring='accuracy')
    
    # ফলাফল প্রদর্শন
    print(f"K-Fold Cross Validation Scores (k={n_folds}):", cv_scores)
    print(f"Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"Standard Deviation: {cv_scores.std():.4f}")

    # নিউরাল নেটওয়ার্ক ট্রেনিং
    print("\n" + "="*50)
    print("Training Neural Network with Selected Features")
    print("="*50)
    model, history = train_neural_network(
        X=X_select.values,  # নির্বাচিত ফিচার
        y=y.values,         # টার্গেট ভেরিয়েবল
        outdir_path=EDA_FIG_PATH,
        batch_size=32,
        epochs=50,
        learning_rate=0.01
    )

    # RandomForest এর পরে XGBoost যোগ করুন
    print("\n" + "="*50)
    print("Training XGBoost with Selected Features")
    print("="*50)

    # XGBoost মডেল তৈরি
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # মডেল ট্রেনিং
    xgb_model.fit(X_train, y_train)

    # প্রেডিকশন
    y_pred_xgb = xgb_model.predict(X_test)

    # মডেল ইভ্যালুয়েশন
    print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))
    print("XGBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
    print("XGBoost Accuracy Score:\n", accuracy_score(y_test, y_pred_xgb))

    # ফিচার ইম্পর্টেন্স প্লট
    plt.figure(figsize=(12, 6))
    xgb_importance = xgb_model.feature_importances_
    indices = np.argsort(xgb_importance)[::-1]
    plt.title('XGBoost Feature Importance')
    plt.bar(range(len(xgb_importance)), xgb_importance[indices], align='center')
    plt.xticks(range(len(xgb_importance)), [list(selected_features)[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f"{EDA_FIG_PATH}/XGBoost_Feature_Importance.jpeg", dpi=300)
    plt.close()

    # XGBoost এর জন্য ক্রস ভ্যালিডেশন
    xgb_cv_scores = cross_val_score(xgb_model, X_select, y, cv=skf, scoring='accuracy')
    print(f"XGBoost K-Fold Cross Validation Scores (k={n_folds}):", xgb_cv_scores)
    print(f"XGBoost Mean Accuracy: {xgb_cv_scores.mean():.4f}")
    print(f"XGBoost Standard Deviation: {xgb_cv_scores.std():.4f}")

    # হাইপারপ্যারামিটার টিউনিং (অপশনাল)
    print("\n" + "="*50)
    print("XGBoost Hyperparameter Tuning")
    print("="*50)

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # সেরা মডেল দিয়ে প্রেডিকশন
    best_xgb_model = grid_search.best_estimator_
    y_pred_best_xgb = best_xgb_model.predict(X_test)

    print("Best XGBoost Classification Report:\n", classification_report(y_test, y_pred_best_xgb))
    print("Best XGBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best_xgb))
    print("Best XGBoost Accuracy Score:\n", accuracy_score(y_test, y_pred_best_xgb))

    # মডেল কম্পেয়ার
    models = {
        'Random Forest': accuracy_score(y_test, y_pred),
        'XGBoost': accuracy_score(y_test, y_pred_xgb),
        'Tuned XGBoost': accuracy_score(y_test, y_pred_best_xgb)
    }

    plt.figure(figsize=(10, 6))
    plt.bar(models.keys(), models.values(), color=['blue', 'green', 'red'])
    plt.title('Model Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.0)
    for i, (model, acc) in enumerate(models.items()):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    plt.tight_layout()
    plt.savefig(f"{EDA_FIG_PATH}/Model_Comparison.jpeg", dpi=300)
    plt.close()

if __name__=="__main__":
    main()

