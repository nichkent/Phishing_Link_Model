# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def main():
    #-------- Load the dataset --------------
    df = pd.read_csv('final_dataset_final.csv', low_memory=False)


    #------- Handle missing/na values in data ---------
    # Separate the features and the target variable
    X = df.drop(columns=['URL', 'label'])
    y = df['label']

    # Identify non-numeric features from the dataset
    non_numeric_features = X.select_dtypes(include=['object']).columns
    print("Non-numeric features:", non_numeric_features.tolist())

    # Handle missing values in non-numeric features and convert to strings for processing
    X[non_numeric_features] = X[non_numeric_features].fillna('Unknown').astype(str)

    # Encode non-numeric features using Label Encoder so the models can learn on them
    for col in non_numeric_features:
        le = LabelEncoder()
        # Fit them to the model
        X[col] = le.fit_transform(X[col])

    # Verify data types after encoding, debugging statements
    #print("Data types after encoding:")
    #print(X.dtypes)

    # Handle missing values in numeric features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    # Fill in na values with the mean
    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].mean())


    #----------- Remove suspect features (possible data leakage) -----------------
    suspect_features = [
        'URLSimilarityIndex', 'DomainTitleMatchScore', 'URLTitleMatchScore',
        'IsResponsive', 'Robots', 'Title', 'HasTitle', 'HasDescription',
        'HasCopyrightInfo', 'HasSocialNet', 'IsHTTPS', 'HasSubmitButton',
        'NoOfJS', 'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL',
        'LetterRatioInURL', 'HasFavicon', 'URLCharProb', 'NoOfImage', 'NoOfExternalRef', 'NoOfCSS', 'Domain', 'NoOfiFrame'
    ]

    # Remove the features from the dataset. There are more than enough features so this is fine
    existing_suspect_features = [feat for feat in suspect_features if feat in X.columns]

    # If a suspect feature is present remove them
    if existing_suspect_features:
        X.drop(columns=existing_suspect_features, inplace=True)
        # Debugging statement
        #print(f"Removed suspect features: {existing_suspect_features}")

    # Check for correlated collumns
    print("Checking for multicollinearity...")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Define a threshold amount of correlation allowed for each column
    high_corr_threshold = 0.8

    # Check each column for correlation
    high_corr_features = [column for column in upper.columns if any(upper[column] > high_corr_threshold)]
    print(f"Features with high correlation (>{high_corr_threshold}): {high_corr_features}")

    # Remove the highly correlated features in order to reduce risk of overfitting
    if high_corr_features:
        X.drop(columns=high_corr_features, inplace=True)
        print(f"Removed highly correlated features: {high_corr_features}")


    #------------ Split data into training and testing sets ----------------
    # Split data into training + validation sets as well as and hold-out test set
    X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Further split training + validation set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
    )
    # Now we have 60% train, 20% validation, 20% hold-out test


    #------------- Feature selection using SelectKBest ---------------------
    print("Starting feature selection with SelectKBest...")
    selector = SelectKBest(f_classif, k=20)

    # Fit the features to the model
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    X_holdout_selected = selector.transform(X_holdout)
    selected_feature_names = X_train.columns[selector.get_support()]

    # Print out the features that were selected by SelectKBest
    print(f"Selected {len(selected_feature_names)} features.")

    #--------------- Run the ensemble model --------------------
    # Initialize individual models (Random Forest, XGBClassifier, and Logistic Regression)
    model1 = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
    )
    model2 = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42
    )
    model3 = LogisticRegression(
        penalty='elasticnet', l1_ratio=0.5, solver='saga',
        max_iter=10000, class_weight='balanced', random_state=42
    )

    # Scale features for the Logistic Regression model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    X_holdout_scaled = scaler.transform(X_holdout_selected)

    # Create a voting classifier with the individual models
    # Use soft voting
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', model1),
            ('xgb', model2),
            ('lr', model3)
        ],
        voting='soft',
        weights=[1, 1, 1],
        n_jobs=-1
    )

    # Fit the ensemble model
    print("Training the ensemble model...")
    ensemble_model.fit(
        np.hstack((X_train_selected, X_train_scaled)), y_train
    )

    # Evaluate on validation set
    print("Evaluating on validation set...")
    y_val_proba = ensemble_model.predict_proba(
        np.hstack((X_val_selected, X_val_scaled))
    )[:, 1]
    y_val_pred = ensemble_model.predict(
        np.hstack((X_val_selected, X_val_scaled))
    )


    #----------------- Results of the model's training ------------------
    # Print out the results of the validation set
    print("Validation Set Classification Report:")
    print(classification_report(y_val, y_val_pred))
    print(f'Validation Set ROC-AUC Score: {roc_auc_score(y_val, y_val_proba):.4f}')

    # Evaluate on hold-out test set
    print("Evaluating on hold-out test set...")
    y_holdout_proba = ensemble_model.predict_proba(
        np.hstack((X_holdout_selected, X_holdout_scaled))
    )[:, 1]
    y_holdout_pred = ensemble_model.predict(
        np.hstack((X_holdout_selected, X_holdout_scaled))
    )

    # Print out the results of the hold out test set
    print("Hold-Out Test Set Classification Report:")
    print(classification_report(y_holdout, y_holdout_pred))
    print(f'Hold-Out Test Set ROC-AUC Score: {roc_auc_score(y_holdout, y_holdout_proba):.4f}')

    # Plot ROC Curve on a graph for the hold out set
    fpr, tpr, thresholds = roc_curve(y_holdout, y_holdout_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_score(y_holdout, y_holdout_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve on Hold-Out Test Set')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    # Plot learning curve on a graph for the Kfold set
    print("Plotting learning curve...")
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5)
    train_sizes, train_scores, val_scores = learning_curve(
        ensemble_model, np.hstack((X_train_selected, X_train_scaled)), y_train,
        cv=skf, scoring='roc_auc',
        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.title('Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('ROC-AUC Score')
    plt.legend()
    plt.show()


    #----------------- Determine highest contributing features -----------------------
    # Analyze feature importances
    print("Analyzing feature importances...")

    # Since ensemble model doesn't have a direct feature_importances_ attribute,
    # we'll average the importances from Random Forest and XGBoost
    rf_importances = model1.fit(X_train_selected, y_train).feature_importances_
    xgb_importances = model2.fit(X_train_selected, y_train).feature_importances_

    avg_importances = (rf_importances + xgb_importances) / 2
    feat_imp_df = pd.DataFrame({'Feature': selected_feature_names, 'Importance': avg_importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
    print("Top features by average importance:")
    print(feat_imp_df.head(10))

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(10))
    plt.title('Top Feature Importances')
    plt.show()

# Call to main function
if __name__ == "__main__":
    main()

