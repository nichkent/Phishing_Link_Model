# Phishing Link Detection Machine Learning Project

### To recreate our findings:
## Download the original dataset from Kaggle.com: 
https://www.kaggle.com/datasets/joebeachcapital/phiusiil-phishing-url
## In a secure environment, run feature_extraction.py.
   Note: For our secure environment, a kali linux vm was created using Oracle Virtual Box. This virtual machine was given its own partition and had any interaction with the host machine disabled including shared folders, clipboards, etc. Afterward, the virtual machine was deleted along with all files. Unfortunately, due to limitations during the study we were unable to run the script on a secure network that was airgapped from our home network which would be advisable in future iterations of the project.
   
## feature_extraction.py results
Make sure that and the PhiUSIIL_Phishing_URL_Dataset.csv are in the same directory as it will need the links from the original dataset to know what to visit. feature_extraction.py will return a csv file called features_dataset.csv with all information it captured during the running of the program.

## Fixing the data
You will notice that much of the data returned by the program are missing values. This is because feature_extraction.py chooses to skip over any features it was not able to find. To fix this, run fix_data.py to fill in all missing values from the original run with 0's for not present. This will return a csv file called final_dataset_final.csv.

## Run advanced_analysis.py
Run advanced_analysis.py to run the ensemble model on the newly collected and now preprocessed dataset. This file will train the model and print out statistics and results relevant to the model's learning.
