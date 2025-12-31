import pandas as pd
import joblib
from sklearn.ensemble import StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load the datasets
file1_path = '../data/processed/merged_datasets/msavi_vari_25_merged.csv'
file2_path = '../data/processed/merged_datasets/ndvi_msavi_25_merged.csv'

dataset1 = pd.read_csv(file1_path)
dataset2 = pd.read_csv(file2_path)

# Load the models
model1_path = 'models/combined_index/wealth/xgboost_msavi_vari_25_25.pkl'
model2_path = 'models/combined_index/wealth/xgboost_ndvi_msavi_25_25.pkl'

model1 = joblib.load(model1_path)
model2 = joblib.load(model2_path)

# Display data structures for validation
dataset1.head(), dataset2.head()

try:
    # Attempt to load models using XGBoost's native method
    model1 = joblib.load(model1_path)
    
    model2 = joblib.load(model2_path)
    
    "Models loaded successfully using XGBoost's native method."
except Exception as e:
    str(e)

# Define the features and target column for both datasets
features_dataset1 = [
    'cluster', 'svyid', 'year', 'country', 'region', 'iso3n', 'households',
    'URBAN_RURA', 'msavi_bin_1', 'msavi_bin_2', 'msavi_bin_3',
    'msavi_bin_4', 'msavi_bin_5', 'msavi_bin_6', 'msavi_bin_7',
    'msavi_bin_8', 'msavi_bin_9', 'msavi_bin_10', 'msavi_bin_11',
    'msavi_bin_12', 'msavi_bin_13', 'msavi_bin_14', 'msavi_bin_15',
    'msavi_bin_16', 'msavi_bin_17', 'msavi_bin_18', 'msavi_bin_19',
    'msavi_bin_20', 'msavi_bin_21', 'msavi_bin_22', 'msavi_bin_23',
    'msavi_bin_24', 'msavi_bin_25', 'vari_bin_1', 'vari_bin_2',
    'vari_bin_3', 'vari_bin_4', 'vari_bin_5', 'vari_bin_6', 'vari_bin_7',
    'vari_bin_8', 'vari_bin_9', 'vari_bin_10', 'vari_bin_11', 'vari_bin_12',
    'vari_bin_13', 'vari_bin_14', 'vari_bin_15', 'vari_bin_16',
    'vari_bin_17', 'vari_bin_18', 'vari_bin_19', 'vari_bin_20',
    'vari_bin_21', 'vari_bin_22', 'vari_bin_23', 'vari_bin_24',
    'vari_bin_25'
]
features_dataset2 = [
    'cluster', 'svyid', 'year', 'country', 'region', 'iso3n', 'households',
    'URBAN_RURA', 'ndvi_bin_1', 'ndvi_bin_2', 'ndvi_bin_3', 'ndvi_bin_4',
    'ndvi_bin_5', 'ndvi_bin_6', 'ndvi_bin_7', 'ndvi_bin_8', 'ndvi_bin_9',
    'ndvi_bin_10', 'ndvi_bin_11', 'ndvi_bin_12', 'ndvi_bin_13',
    'ndvi_bin_14', 'ndvi_bin_15', 'ndvi_bin_16', 'ndvi_bin_17',
    'ndvi_bin_18', 'ndvi_bin_19', 'ndvi_bin_20', 'ndvi_bin_21',
    'ndvi_bin_22', 'ndvi_bin_23', 'ndvi_bin_24', 'ndvi_bin_25',
    'msavi_bin_1', 'msavi_bin_2', 'msavi_bin_3', 'msavi_bin_4',
    'msavi_bin_5', 'msavi_bin_6', 'msavi_bin_7', 'msavi_bin_8',
    'msavi_bin_9', 'msavi_bin_10', 'msavi_bin_11', 'msavi_bin_12',
    'msavi_bin_13', 'msavi_bin_14', 'msavi_bin_15', 'msavi_bin_16',
    'msavi_bin_17', 'msavi_bin_18', 'msavi_bin_19', 'msavi_bin_20',
    'msavi_bin_21', 'msavi_bin_22', 'msavi_bin_23', 'msavi_bin_24',
    'msavi_bin_25'
]
target_column = 'wealth'

# Extract features and target
X1 = dataset1[features_dataset1]
y1 = dataset1[target_column]

X2 = dataset2[features_dataset2]
y2 = dataset2[target_column]

# Display data readiness
print(X1.shape, X2.shape, y1.shape, y2.shape)


from sklearn.preprocessing import LabelEncoder

X1 = pd.get_dummies(X1)
X2 = pd.get_dummies(X2)

# Handle missing values
imputer1 = SimpleImputer(strategy='mean')
X1 = imputer1.fit_transform(X1)

# Handle missing values
imputer2 = SimpleImputer(strategy='mean')
X2 = imputer2.fit_transform(X2)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

meta_model = Ridge()

# Create the stacking regressor
stacking_reg = StackingRegressor(estimators=[("model1", model1), ("model2", model2)], final_estimator=meta_model)

# Fit the stacking regressor on the training data
stacking_reg.fit(X1_train, y1_train)

# Predict using the stacking regressor
ensemble_predictions = stacking_reg.predict(X1_test)

# Retry predictions and ensemble
#predictions1 = model1.predict(X1)
#predictions2 = model2.predict(X2)

#ensemble_predictions = (predictions1 + predictions2) / 2

# Compute the R² score for the ensemble model
r2_ensemble = r2_score(y1_test, ensemble_predictions)
print(r2_ensemble)
