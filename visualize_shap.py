from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd

df_path = "../data/processed/NDVI/ndvi_merged_10_bins.csv"
bins_count = 10
# Load the merged dataset
merged_df = pd.read_csv(df_path)
bins_columns_range = (bins_count) * -1

# List of bin column names
original_bin_columns = merged_df.columns[bins_columns_range:].tolist()

# Create a mapping of original bin columns to new bin_<i> names
new_bin_columns = {original_bin_columns[i]: f'bin_{i+1}' for i in range(len(original_bin_columns))}

# Rename the bin columns in the DataFrame
merged_df.rename(columns=new_bin_columns, inplace=True)

# Select relevant features including bin columns and other columns
features = merged_df[['cluster','svyid','year','iso3n', 'country', 'region', 'households', 'URBAN_RURA'] + [f'bin_{i+1}' for i in range(len(original_bin_columns))]]

# Save feature names
target = merged_df['wealthpooled']

# Drop rows where the target column (wealth) has NaN values
features = features[~target.isna()]
target = target.dropna()

# Convert categorical features to numeric
features = pd.get_dummies(features)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
features = imputer.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model_path = 'models/individual_index/wealthpooled/ndvi_random_forest_bins_10.pkl'

feature_names = ['cluster','svyid','year','iso3n', 'country', 'region', 'households', 'URBAN_RURA']
model = joblib.load(model_path)

# 1. Create a SHAP explainer for a tree model
explainer = shap.TreeExplainer(model)   # model = your RF or XGBoost model

# 2. Compute SHAP values
shap_values = explainer.shap_values(X_train)  # X must be the SAME feature set used to train

# 3. Plot summary with feature names
shap.summary_plot(shap_values, X_train, feature_names=X_train.columns)
