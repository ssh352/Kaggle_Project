# Kaggle_Project
For All-state Kaggle Project

Workflow for stacking:  
for every model:
  1. load the processed data (processed.zip)
  2. tune parameters using train.csv (cross-validate)
  2. train model on train.csv
  3. train model using the same parameters on train_full.csv
  4. predict test.csv
  5. predict submit.csv
  6. output two predictions
