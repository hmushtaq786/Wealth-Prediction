import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

# Paths to datasets
dataset_1_path = "../data/processed/merged_datasets/ndvi_vari_20_merged.csv"
dataset_2_path = "../data/processed/merged_datasets/msavi_vari_20_merged.csv"

# Paths to models directory
models_directory = "models/combined_index/wealth"

# Load datasets
dataset_1 = pd.read_csv(dataset_1_path)
dataset_2 = pd.read_csv(dataset_2_path)

# Define unused columns
unused_columns = [
    "wealthpooled", "wealthpooled5country", "wealth", "iso3", "hv000",
    "cname", "LATNUM", "LONGNUM", "msavi_mean", "vari_mean", 
    "ndvi_mean", "ndmi_mean", "ndbi_mean", "mndwi_mean"
]

# Separate features and target
X1 = dataset_1.drop(columns=unused_columns, errors='ignore')
y1 = dataset_1["wealth"]

X2 = dataset_2.drop(columns=unused_columns, errors='ignore')
y2 = dataset_2["wealth"]

# Apply one-hot encoding (if needed) for categorical columns
X1_dummies = pd.get_dummies(X1)
X2_dummies = pd.get_dummies(X2)

# Align columns (union of all columns from both datasets)
all_columns = X1_dummies.columns.union(X2_dummies.columns)

# Reindex to ensure both datasets have the same columns
X1_aligned = X1_dummies.reindex(columns=all_columns, fill_value=0)
X2_aligned = X2_dummies.reindex(columns=all_columns, fill_value=0)

# Split into training and validation subsets
X1_train, X1_val, y1_train, y1_val = train_test_split(X1_aligned, y1, test_size=0.2, random_state=42)
X2_train, X2_val, y2_train, y2_val = train_test_split(X2_aligned, y2, test_size=0.2, random_state=42)

# Combine training and validation subsets
X_train = pd.concat([X1_train, X2_train], ignore_index=True)
y_train = pd.concat([y1_train, y2_train], ignore_index=True)

X_val = pd.concat([X1_val, X2_val], ignore_index=True)
y_val = pd.concat([y1_val, y2_val], ignore_index=True)

# Load models
models = {
    'xgb_model_1': joblib.load(f'{models_directory}/xgboost_msavi_vari_25_25.pkl'),
    'xgb_model_2': joblib.load(f'{models_directory}/xgboost_ndvi_msavi_25_25.pkl'),
    'rf_model_1': joblib.load(f'{models_directory}/random_forest_msavi_vari_25_25.pkl'),
    'rf_model_2': joblib.load(f'{models_directory}/random_forest_ndvi_vari_25_25.pkl'),
}

# Ensure validation data matches training data columns
X_val_aligned = X_val.reindex(columns=X_train.columns, fill_value=0)

# Check alignment
print(f"Validation dataset shape: {X_val_aligned.shape}")
print(f"Training dataset feature count: {X_train.shape[1]}")

# Generate predictions
predictions = {name: model.predict(X_val_aligned) for name, model in models.items()}

# Ensembling Techniques

# 1. Simple Averaging
ensemble_preds_avg = np.mean(list(predictions.values()), axis=0)
mse_avg = mean_squared_error(y_val, ensemble_preds_avg)
r2_avg = r2_score(y_val, ensemble_preds_avg)

# 2. Weighted Averaging
weights = {'xgb_model_1': 0.4, 'xgb_model_2': 0.3, 'rf_model_1': 0.2, 'rf_model_2': 0.1}
ensemble_preds_weighted = np.sum([weights[name] * preds for name, preds in predictions.items()], axis=0)
mse_weighted = mean_squared_error(y_val, ensemble_preds_weighted)
r2_weighted = r2_score(y_val, ensemble_preds_weighted)

# 3. Stacking
stacking_features = np.column_stack(list(predictions.values()))
meta_model = Ridge()
meta_model.fit(stacking_features, y_val)
stacked_preds = meta_model.predict(stacking_features)
mse_stacked = mean_squared_error(y_val, stacked_preds)
r2_stacked = r2_score(y_val, stacked_preds)

# Save results
results = pd.DataFrame({
    "Method": ["Simple Averaging", "Weighted Averaging", "Stacking"],
    "MSE": [mse_avg, mse_weighted, mse_stacked],
    "R2": [r2_avg, r2_weighted, r2_stacked]
})

results.to_csv("ensemble_results.csv", index=False)

# Print results
print("Ensemble Results:")
print(results)
