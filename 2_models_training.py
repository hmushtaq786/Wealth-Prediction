import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import glob  # For reading multiple files

def model_training(bins_count, model_name, df_path, index_name):
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

    # Initialize the regressor
    model = XGBRegressor(random_state=42)
    param_grid = {}
    if model_name == "xgboost":
        model = XGBRegressor(random_state=42)
        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5, 7]
        } 
    elif model_name == "gradient_boost":
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5, 7]
        } 
    elif model_name == "random_forest":
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20]
        } 

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Predict on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model using R² and MSE
    test_r2 = r2_score(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)

    # print(f"Model: {model_name}, Bins: {bins_count}, Test R²: {test_r2:.4f}, Test MSE: {test_mse:.4f}")

    # Update results_dict for the current bin count
    results_dict = {
        'Model': model_name,
        'Bins': bins_count,
        'Index': index_name,
        'Best Parameters': grid_search.best_params_,
        'Test R²': test_r2,
        'Test MSE': test_mse
    }

    # Save the best model for future use
    joblib.dump(best_model, f'models/combined_index/wealthpooled/{index_name}_{model_name}_bins_{bins_count}.pkl')

    return results_dict

bins_array = [10, 15, 20, 25, 30]
models = ["xgboost", "gradient_boost", "random_forest"]
indices = ["ndvi", "ndmi", "msavi", "ndbi", "vari", "mndwi"]

results = []
# Train models and collect results
for index in tqdm(indices, desc="Loop for each index"):
    for model in tqdm(models, desc="Loop for each model"):
        for bins in tqdm(bins_array, desc="Loop for each bin"):
            # print(f"{model}: bin - {bins}")
            df_path= f"../data/processed/{index.upper()}/{index}_merged_{bins}_bins.csv"
            results_dict = model_training(bins, model, df_path, index)
            results.append(results_dict)

df = pd.DataFrame(results) 
df.to_csv("model_results_combined_wealthpooled.csv", index=False)