# test.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

def main():
    # Load the dataset
    df = pd.read_csv('final_dataset.csv')

    # Separate features and target variable
    X = df.drop(columns=['URL', 'label'])
    y = df['label']

    # Identify non-numeric features
    non_numeric_features = X.select_dtypes(include=['object']).columns
    print("Non-numeric features:", non_numeric_features.tolist())

    # Handle missing values in non-numeric features
    X[non_numeric_features] = X[non_numeric_features].fillna('Unknown')

    # Encode non-numeric features using Label Encoder
    for col in non_numeric_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Verify data types after encoding
    print("Data types after encoding:")
    print(X.dtypes)

    # Handle missing values in numeric features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].mean())

    # Check for duplicate URLs
    duplicate_urls = df['URL'].duplicated().sum()
    print(f"Number of duplicate URLs: {duplicate_urls}")

    if duplicate_urls > 0:
        # Remove duplicate URLs
        df.drop_duplicates(subset='URL', inplace=True)
        X = df.drop(columns=['URL', 'label'])
        y = df['label']

        # Re-encode non-numeric features after dropping duplicates
        non_numeric_features = X.select_dtypes(include=['object']).columns
        for col in non_numeric_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    # Remove suspect features (possible data leakage)
    suspect_features = [
        'URLSimilarityIndex', 'DomainTitleMatchScore', 'URLTitleMatchScore',
        'IsResponsive', 'Robots', 'Title', 'HasTitle', 'HasDescription',
        'HasCopyrightInfo', 'NoOfSelfRef', 'NoOfEmptyRef', 'NoOfExternalRef',
        # Features identified as too predictive
        'IsHTTPS', 'HasSocialNet', 'url_length', 'HasSubmitButton',
        'NoOfJS', 'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL',
        'LetterRatioInURL', 'HasFavicon', 'URLCharProb'
    ]
    existing_suspect_features = [feat for feat in suspect_features if feat in X.columns]
    if existing_suspect_features:
        X.drop(columns=existing_suspect_features, inplace=True)
        print(f"Removed suspect features: {existing_suspect_features}")

    # Check for multicollinearity
    print("Checking for multicollinearity...")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper.columns if any(upper[column] > 0.8)]
    print(f"Features with high correlation (>0.8): {high_corr_features}")

    # Optionally, remove highly correlated features
    if high_corr_features:
        X.drop(columns=high_corr_features, inplace=True)
        print(f"Removed highly correlated features: {high_corr_features}")

    # Split data into training and test sets before scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Check for overlapping indices
    overlap = set(X_train.index).intersection(set(X_test.index))
    print(f"Number of overlapping samples between training and test sets: {len(overlap)}")

    # Feature scaling on training and test sets separately
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Check class distribution
    print("Class distribution in the dataset:")
    print(y.value_counts())
    print("Class distribution in training set:")
    print(y_train.value_counts())
    print("Class distribution in test set:")
    print(y_test.value_counts())

    # Initialize the logistic regression model with regularization
    logreg_model = LogisticRegression(
        penalty='elasticnet', l1_ratio=0.5, solver='saga', max_iter=5000, C=0.01, random_state=42
    )

    # Create a pipeline for cross-validation with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', logreg_model)
    ])

    # Perform Stratified K-Fold Cross-Validation
    print("Performing Stratified K-Fold Cross-Validation with Elastic Net regularization...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc')
    print(f'Cross-validation ROC-AUC scores: {cv_scores}')
    print(f'Mean ROC-AUC score: {cv_scores.mean():.4f}')

    # Train the model
    logreg_model.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = logreg_model.predict(X_test_scaled)
    y_proba = logreg_model.predict_proba(X_test_scaled)[:, 1]

    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f'ROC-AUC Score on test set: {roc_auc_score(y_test, y_proba):.4f}')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Analyze coefficients
    feature_names = X_train.columns  # Ensure we have feature names
    coefficients = logreg_model.coef_[0]

    # Create DataFrame for coefficients
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)
    print("Top 10 features by absolute coefficient value:")
    print(coef_df.head(10))

    # Plot feature coefficients
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(10))
    plt.title('Top 10 Feature Coefficients')
    plt.show()

if __name__ == "__main__":
    main()
