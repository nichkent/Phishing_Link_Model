# Imports
import pandas as pd

def main():
    # Read the scraped features dataset (1,000 URLs)
    features_df = pd.read_csv('features_dataset.csv')

    # Read the original dataset (over 200,000 entries)
    original_df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')

    # List of features from the original dataset to include
    original_features = ['URL', 'URLLength', 'Domain', 'DomainLength', 'IsDomainIP', 'TLD', 'URLSimilarityIndex',
                         'CharContinuationRate', 'TLDLegitimateProb', 'URLCharProb', 'TLDLength', 'NoOfSubDomain',
                         'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio', 'NoOfLettersInURL',
                         'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL', 'NoOfEqualsInURL', 'NoOfQMarkInURL',
                         'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS',
                         'LineOfCode', 'LargestLineLength', 'HasTitle', 'Title', 'DomainTitleMatchScore',
                         'URLTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive', 'NoOfURLRedirect',
                         'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup', 'NoOfiFrame', 'HasExternalFormSubmit',
                         'HasSocialNet', 'HasSubmitButton', 'HasHiddenFields', 'HasPasswordField', 'Bank', 'Pay',
                         'Crypto', 'HasCopyrightInfo', 'NoOfImage', 'NoOfCSS', 'NoOfJS', 'NoOfSelfRef', 'NoOfEmptyRef',
                         'NoOfExternalRef']

    # Select only the required features from the original dataset
    original_selected_df = original_df[original_features]

    # Merge the datasets on 'URL', keeping only the 1,000 scraped URLs
    merged_df = pd.merge(features_df, original_selected_df, on='URL', how='left')


    # Handle the duplicate columns from merging
    if 'label_y' in merged_df.columns:
        merged_df.drop(columns=['label_y'], inplace=True)
        merged_df.rename(columns={'label_x': 'label'}, inplace=True)

    # Save the final dataset
    merged_df.to_csv('final_dataset.csv', index=False)
    print("Final dataset saved to final_dataset.csv")

# Call the main function
if __name__ == "__main__":
    main()

