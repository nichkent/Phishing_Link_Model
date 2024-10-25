# Phishing Link Detection Machine Learning Project

### To recreate our findings:
1. Download the original dataset from Kaggle.com: https://www.kaggle.com/datasets/joebeachcapital/phiusiil-phishing-url
2. In a secure environment, run feature_extraction.py.
   Note: For our secure environment, a kali linux vm was created using Oracle Virtual Box. This virtual machine was given its own partition had any interaction with the host machine disabled including shared folders, clipboards, etc. Afterwards, the virtual machine was deleted along with all files. Unfortunately, due to limitations during the study we were unable to run the script on a secure network that was airgapped from our home network which would be advisable in future iterations of the project.
3. Once completed, the script will output a file named features_dataset.csv. With the PhiUSIIL_Phising_URL_Dataset.csv and features_dataset.csv files in the same directory, execute combine_scripts.py.
4. After running the script, a file titled final_dataset.csv will be created which is the combination of the features we found from our work as well as the features from the original dataset on 1000 randomly sampled URLs.
5. Run main.py, ensuring final_dataset.csv is within the same directory. This will print the classification report as well as the ROC_AUC score to the terminal.