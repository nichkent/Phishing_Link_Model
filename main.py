# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the finalized dataset
df = pd.read_csv('final_dataset.csv')

# Define features and target variable
X = df[['num_subdomains', 'is_ip_address', 'uses_https', 'has_at_symbol',
        'has_hyphen_in_domain', 'num_digits_in_url', 'double_slash_redirect', 'https_in_url', 'uses_shortening_service',
        'num_script_tags', 'num_iframe_tags', 'num_embed_tags', 'num_object_tags', 'num_form_tags', 'num_anchor_tags',
        'num_external_resources', 'external_forms', 'has_eval', 'has_escape', 'has_unescape',
        'has_exec', 'uses_document_cookie', 'uses_window_location', 'uses_settimeout', 'uses_setinterval', 'uses_prompt',
        'uses_alert', 'uses_confirm', 'num_meta_tags', 'has_meta_refresh', 'has_meta_keywords', 'has_meta_description',
        'num_images', 'num_external_images', 'num_data_images', 'ssl_certificate_issuer', 'is_cert_verified',
        'has_copyright', 'content_length', 'num_links', 'num_external_links', 'num_internal_links', 'num_empty_links',
        'has_favicon', 'has_login_form', 'has_social_media_links', 'has_meta_redirect', 'num_emails_in_page',
        'page_title_length', 'DomainLength', 'IsDomainIP', 'TLD', 'URLSimilarityIndex',
        'CharContinuationRate', 'TLDLegitimateProb', 'URLCharProb', 'TLDLength', 'NoOfSubDomain', 'HasObfuscation',
        'NoOfObfuscatedChar', 'ObfuscationRatio', 'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL',
        'DegitRatioInURL', 'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL',
        'SpacialCharRatioInURL', 'IsHTTPS', 'LineOfCode', 'LargestLineLength', 'HasTitle', 'Title', 'DomainTitleMatchScore',
        'URLTitleMatchScore', 'Robots', 'IsResponsive', 'NoOfURLRedirect', 'NoOfSelfRedirect', 'HasDescription',
        'NoOfPopup', 'NoOfiFrame', 'HasExternalFormSubmit', 'HasSocialNet', 'HasSubmitButton', 'HasHiddenFields',
        'HasPasswordField', 'Bank', 'Pay', 'Crypto', 'HasCopyrightInfo', 'NoOfImage', 'NoOfCSS', 'NoOfJS',
        'NoOfSelfRef', 'NoOfEmptyRef', 'NoOfExternalRef']]
y = df['label']

# Identify non-numeric features
non_numeric_features = X.select_dtypes(include=['object']).columns
print("Non-numeric features:", non_numeric_features)

# Handle missing values in non-numeric features (fill with 'Unknown')
X[non_numeric_features] = X[non_numeric_features].fillna('Unknown')

# Encode non-numeric features using Label Encoder
le = LabelEncoder()
for col in non_numeric_features:
    X[col] = le.fit_transform(X[col])

# Handle any remaining missing values in numeric features
# For simplicity, fill numeric missing values with the mean of the column
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
X[numeric_features] = X[numeric_features].fillna(X[numeric_features].mean())

# Remove suspect features
features_to_remove = [
    'URLSimilarityIndex', 'DomainTitleMatchScore', 'URLTitleMatchScore',
    'IsResponsive', 'Robots', 'Title'
]
X_cleaned = X.drop(columns=features_to_remove)

# Proceed with preprocessing on X_cleaned

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the logistic regression model with regularization
logreg_model = LogisticRegression(
    penalty='l1', solver='saga', max_iter=1000, C=0.1, random_state=42
)

# Train the model
logreg_model.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg_model.predict(X_test)
y_proba = logreg_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print(classification_report(y_test, y_pred))
print(f'ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}')
