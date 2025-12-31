import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset (replace with your dataset path)
dataset_path = '../data/processed/merged_datasets/msavi_vari_25_merged.csv'
dataset = pd.read_csv(dataset_path)

# Define features and target column (adjust column names as necessary)
features = [
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
target = 'wealthpooled'  # Target column

# Extract features and target
X = dataset[features]

# Identify categorical columns
categorical_columns = ['svyid', 'country', 'region', 'iso3n', 'URBAN_RURA']  # Replace with your actual categorical column names

# Apply One-Hot Encoding
column_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)
    ],
    remainder='passthrough'  # Keep other (numerical) columns unchanged
)

X = column_transformer.fit_transform(X)

y = dataset[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for XGBoost and Random Forest)
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train Random Forest model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# gb_model.fit(X_train, y_train)

# Ensemble using Stacking
meta_model = Ridge()  # You can experiment with other models like XGBoost, LinearRegression, etc.
stacking_model = StackingRegressor(
    estimators=[('xgb', xgb_model), ('rf', rf_model)],
    final_estimator=meta_model

)

# Fit the stacking model
stacking_model.fit(X_train, y_train)

# Predictions
xgb_predictions = xgb_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
# gb_predictions = gb_model.predict(X_test)
stacking_predictions = stacking_model.predict(X_test)

# Evaluate the models using R² score
r2_xgb = r2_score(y_test, xgb_predictions)
r2_rf = r2_score(y_test, rf_predictions)
# r2_gb = r2_score(y_test, gb_predictions)
r2_stacking = r2_score(y_test, stacking_predictions)

# Print R² scores
print(f"R² score for XGBoost: {r2_xgb:.4f}")
print(f"R² score for Random Forest: {r2_rf:.4f}")
# print(f"R² score for Gradient Boost: {r2_gb:.4f}")
print(f"R² score for Stacking Regressor: {r2_stacking:.4f}")

# Simple Ensemble (Averaging Predictions)
ensemble_predictions = (xgb_predictions + rf_predictions) / 2
r2_ensemble = r2_score(y_test, ensemble_predictions)
print(f"R² score for Ensemble (Averaging): {r2_ensemble:.4f}")