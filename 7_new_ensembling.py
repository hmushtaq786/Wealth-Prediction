import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Load the models from pkl files
with open('models/combined_index/wealthpooled/xgboost_msavi_vari_25_25.pkl', 'rb') as file:
    xgb_model_1 = joblib.load(file)

with open('models/combined_index/wealthpooled/xgboost_ndvi_ndbi_25_25.pkl', 'rb') as file:
    xgb_model_2 = joblib.load(file)

with open('models/combined_index/wealthpooled/random_forest_msavi_vari_25_25.pkl', 'rb') as file:
    rf_model_1 = joblib.load(file)

with open('models/combined_index/wealthpooled/random_forest_ndvi_ndbi_25_25.pkl', 'rb') as file:
    rf_model_2 = joblib.load(file)


df = pd.read_csv("../data/processed/merged_datasets/ndvi_msavi_25_merged.csv")
bins_count = 50
bins_columns_range = (bins_count + 1) * -1

# List of bin column names
original_bin_columns = df.columns[bins_columns_range:-1].tolist()
# Create a mapping of original bin columns to new bin_<i> names
new_bin_columns = {original_bin_columns[i]: f'bin_{i+1}' for i in range(len(original_bin_columns))}

# Rename the bin columns in the DataFrame
df.rename(columns=new_bin_columns, inplace=True)

# Select relevant features including bin columns and other columns
features = df[['cluster','svyid','year','iso3n', 'country', 'LATNUM', 'LONGNUM', 'region', 'households', 'URBAN_RURA'] + [f'bin_{i+1}' for i in range(len(original_bin_columns))]]

target = df['wealth']

# Drop rows where the target column (wealth) has NaN values
features = features[~target.isna()]
target = target.dropna()

# Convert categorical features to numeric
features = pd.get_dummies(features)

imputer = SimpleImputer(strategy='most_frequent')
features = imputer.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# Assume X_test and y_test are your test data and labels
# X_test = ...
# y_test = ...

# Create a voting classifier
# voting_clf = VotingRegressor(estimators=[('xgb1', xgb_model_1), ('xgb2', xgb_model_2)])

# Train the voting classifier if necessary (usually stacking/blending requires this)
# voting_clf.fit(X_train, y_train)

# Predict using the voting classifier
# y_pred = voting_clf.predict(X_test)

# Evaluate the accuracy
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f'Ensemble model Mean Squared Error: {mse}')
# print(f'Ensemble model R2 score: {r2}')


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

total_models = {
    "xgboost_i_msavi_25_vari_25": xgb_model_1, 
    "xgboost_i_ndvi_25_ndbi_25": xgb_model_2, 
    "rf_i_msavi_25_vari_25": rf_model_1, 
    "rf_i_ndvi_25_ndbi_25": rf_model_2, 

    }

df_res = pd.DataFrame(columns=['model_combination', 'index', 'MSE', 'R2'])

for model_name_1, model_1 in total_models.items():
    for model_name_2, model_2 in total_models.items():
        if model_name_1 == model_name_2:
            continue

        combination_name = f"{model_name_1.split('_i_')[0]}_and_{model_name_2.split('_i_')[0]}"
        index = f"{model_name_1.split('_i_')[1]}_and_{model_name_2.split('_i_')[1]}"
    
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        meta_model = Ridge()

        # Create the stacking regressor
        stacking_reg = StackingRegressor(estimators=[(model_name_1, model_1), (model_name_2, model_2)], final_estimator=meta_model)

        # Fit the stacking regressor on the training data
        stacking_reg.fit(X_train, y_train)

        # Predict using the stacking regressor
        y_pred = stacking_reg.predict(X_test)

        # Evaluate the performance using mean squared error (or any other regression metric)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        result = {
            "model_combination": combination_name,
            "index": index,
            "MSE": mse,
            "R2": r2,
        }
        df_res = pd.concat([df_res, pd.DataFrame([result])], ignore_index=True)
        print(f'{combination_name} Stacking Ensemble Model Mean Squared Error: {mse}')
        print(f'{combination_name} Stacking Ensemble Model R2 score: {r2}')

df_res.to_csv("ensembled_models_results_wealth_new.csv", index=False)