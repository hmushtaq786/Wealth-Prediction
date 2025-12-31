import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File path
file_path = "models/individual_index/wealthpooled/ndvi_xgboost_bins_25.pkl"  # Replace with your actual file path

# Load data
data_path = "model_results_wealthpooled.csv"  # Replace with your dataset path
data = pd.read_csv(data_path)

# Display general information about the dataset
print("\nDataset Overview:")
print(data.info())

# Summary Statistics
print("\nSummary Statistics:")
print(data.describe())

# Filter numeric columns only for correlation
numeric_data = data.select_dtypes(include=["number"])

# Compute correlation matrix
correlation_matrix = numeric_data.corr()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

# If you see a weak or moderate correlation, it may warrant further investigation to understand the nature of the relationship.
# Consider visualizing scatterplots of these variables for additional insights into their interaction. For example:
sns.scatterplot(
    x='Bins', 
    y='Test R²', 
    hue='Model', 
    style='Index', 
    data=data, 
    palette='tab10', 
    s=70  # Set marker size (default is smaller, e.g., ~40)
)

plt.title("Scatterplot of Bins vs Test R² (Fixed Marker Size)", fontsize=16)
plt.xlabel("Bins", fontsize=14)
plt.ylabel("Test R²", fontsize=14)
plt.legend(title="Legend", loc='best', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# Feature Importance
import joblib

# Load the saved model
model = joblib.load(file_path)

# Load training data to get feature names
training_data_path = "../data/processed/NDVI/ndvi_merged_25_bins.csv"  # Replace with your dataset path
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
    feature_importance = model.feature_importances_
elif hasattr(model, "get_booster"):
    booster = model.get_booster()
    importance_dict = booster.get_score(importance_type="weight")
    feature_names = list(importance_dict.keys())
    feature_importance = list(importance_dict.values())
else:
    raise AttributeError("Model does not support feature importance extraction.")

print(f"Length of feature_names: {len(feature_names)}")
print(f"Length of feature_importance: {len(feature_importance)}")

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

# Display top 10 features
print("\nTop 10 Features by Importance:")
print(importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(30), palette="viridis")
plt.title("Top 30 Features by Importance", fontsize=16)
plt.xlabel("Importance Score", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.tight_layout()
plt.show()

# Comparing Performance Across Bins
bins = sorted(data['Bins'].unique())  # Replace 'bin_column_name' with your actual column
metrics = ['Test R²', 'Test MSE']  # Replace with your relevant metrics

for metric in metrics:
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=data, x='Bins', y=metric, ci=None, palette="husl")  # Replace with your column names
    plt.title(f"{metric} Across Bins", fontsize=16)
    plt.xlabel("Bins", fontsize=14)
    plt.ylabel(metric, fontsize=14)
    ax.axis(ymin=0.6,ymax=0.7)
    plt.tight_layout()
    plt.show()

# Comparing Models (if applicable)
if 'Model' in data.columns:  # Replace with your actual column
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=data, x='Model', y=metric, hue='Bins', palette="coolwarm")
        plt.title(f"{metric} by Model and Bin", fontsize=16)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.legend(title="Bins")
        ax.axis(ymin=0.6,ymax=0.7)
        plt.tight_layout()
        plt.show()
