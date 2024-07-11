import os


import pandas as pd
from xgb_process import shap_summary,xgb_model
import yaml
import xgboost as xgb
import json
import numpy as np
import pickle
from create_dice_models import DiceModelCreator

##Load config file named conf.yml
with open('conf_telchurn.yml') as file:
    conf = yaml.load(file, Loader=yaml.FullLoader)

#Load processed data
df=pd.read_csv(conf['data']['processed_data_path'])
cat_cols=conf['model']['features']['cat_features']
num_cols=conf['model']['features']['num_features']
target=conf['model']['features']['target']
id_features=conf['model']['features']['id_features']


# Initialize and use the model
model_instance = xgb_model.XGBoostModel(df=df, cat_features=cat_cols, num_features=num_cols, target=target,id_features=id_features, mode='dev')
model,dtrain,X_train,dtest,X_test=model_instance.train_model()
X_train=X_train.reset_index(drop=True)
X_test=X_test.reset_index(drop=True)

print("Saving model")
# Save the model to a pickle file
with open(conf['model']['model_location'], 'wb') as f:
    pickle.dump(model, f)


print("Save train and test data with predictions")
X_train.to_csv(conf['model']['predicted_train_data'], index=False)
X_test.to_csv(conf['model']['predicted_test_data'], index=False)


analyzer=shap_summary.ShapAnalyzer(model=model,
                                X_train=X_train,
                                dtrain=dtrain,
                                cat_features=cat_cols,
                                num_features=num_cols)

##Save the local SHAP explanation for X_test and X_train
print("Getting Local SHAP explanation saved")
explainer=analyzer.get_explainer()

shap_train=pd.DataFrame(explainer.shap_values(xgb.DMatrix(X_train[model.feature_names], enable_categorical=True)),columns=model.feature_names)
shap_train[id_features]=X_train[id_features]

shap_test=pd.DataFrame(explainer.shap_values(xgb.DMatrix(X_test[model.feature_names], enable_categorical=True)),columns=model.feature_names)
shap_test[id_features]=X_test[id_features]

print("Saving SHAP values")
shap_train.columns = ['shapvalue_' + col if col != 'customerid' else col for col in shap_train.columns]
shap_test.columns = ['shapvalue_' + col if col != 'customerid' else col for col in shap_test.columns]
shap_train.to_csv(conf['model']['train_shap_values'], index=False)
shap_test.to_csv(conf['model']['test_shap_values'], index=False)

print("Saving Big Query Import Tables")
bq_data=pd.concat([X_train,X_test],axis=0)
bq_shap_data=pd.concat([shap_train,shap_test],axis=0)
bq_data.to_csv(conf['model']['bq_import_data'],index=False)
bq_shap_data.to_csv(conf['model']['bq_import_shap_data'],index=False)

print("Get SHAP summary and results")
result_df = analyzer.analyze_shap_values()
summary_df = analyzer.summarize_shap_df()

# Convert object types to category and numeric types to their respective numeric types
for col in summary_df.columns:
    if summary_df[col].dtype == 'object':
        summary_df[col] = summary_df[col].astype('category')
    elif pd.api.types.is_numeric_dtype(summary_df[col]):
        summary_df[col] = pd.to_numeric(summary_df[col])

summary_df=summary_df.sort_values(by=['Feature_Importance_Rank','feature_group','probability_contribution'],ascending=[True,True,True]).reset_index(drop=True)
# Convert the 'Importance Rank' and 'Group' columns to a consistent data type before sorting
result_df['Importance Rank'] = result_df['Importance Rank'].astype(str)
result_df['Group'] = result_df['Group'].astype(str)
result_df=result_df.sort_values(by=['Importance Rank','Group'], ascending=[True, True])




print("Saving summary and result DataFrames to CSV files")
# Save the summary and result DataFrames to CSV files
result_df=result_df[['Feature','Group', 'SHAP Value', 'Adjusted Probability', 'Probability Change (%)'
       , 'Feature Importance', 'Importance Rank']]
result_df.columns=['Feature','Feature_Subset', 'SHAP_Value', 'Adjusted_Probability', 'Probability_Change_Perc'
       , 'Feature_Importance', 'Importance_Rank']

summary_df.to_csv(conf['model']['shap_summary'], index=False)
result_df.to_csv(conf['model']['shap_results'], index=False)


print("Creating and Saving DiCE model and base histrgram model")
# Create an instance of the DiceModelCreator class
model_creator = DiceModelCreator(conf)
model_creator.save_models()


