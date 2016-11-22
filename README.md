# Kaggle_Project
For All-state Kaggle Project

Workflow for stacking:  
for every model:
  1. use the processed data (processed.zip)
  2. tune parameters using train.csv (cross-validate)
  2. train model on train.csv (model 1)
  3. train model using the same parameters on train_full.csv (model 2)
  4. predict test.csv using model 1
  5. predict submit.csv using model 2
  6. output two predictions
