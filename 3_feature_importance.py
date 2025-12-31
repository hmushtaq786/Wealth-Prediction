import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_file_path = "models/xgboost_bins_10.pkl"  # Replace with your actual model file path
model = joblib.load(model_file_path)

# Load training data to get feature names
training_data_path = "../data/processed/NDVI/ndvi_merged_10_bins.csv"  # Replace with your dataset path
merged_df = pd.read_csv(training_data_path)

# Extract relevant feature names (modify to match preprocessing)
bins_columns_range = -10  # Adjust for the number of bins used
original_bin_columns = merged_df.columns[bins_columns_range:].tolist()
new_bin_columns = {original_bin_columns[i]: f'bin_{i+1}' for i in range(len(original_bin_columns))}
merged_df.rename(columns=new_bin_columns, inplace=True)
features = pd.get_dummies(merged_df[['country', 'region', 'households', 'URBAN_RURA'] + list(new_bin_columns.values())])
feature_names = features.columns  # Ensure feature_names is always defined

# Extract feature importance
if hasattr(model, "feature_importances_"):
    # For Random Forest and Gradient Boosting models
    feature_importance = model.feature_importances_
elif hasattr(model, "get_booster"):  # For XGBoost
    # Extract feature importance from the booster
    booster = model.get_booster()
    importance_dict = booster.get_score(importance_type="weight")
    feature_names = list(importance_dict.keys())  # Override feature_names if using booster
    feature_importance = list(importance_dict.values())
else:
    raise AttributeError("Model does not support feature importance extraction.")

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

# Display the top features
print("Top Contributing Features:")
print(importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(20), palette="viridis")
plt.title("Top 20 Feature Contributions", fontsize=16)
plt.xlabel("Importance Score", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.tight_layout()
plt.show()
