import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

def model_train(model_name, merged_df, index_name):

    bin_count = int(index_name.split("_")[-1])

    index1 = index_name.split("_")[0]
    index2 = index_name.split("_")[1]

    features = merged_df[
            	# ['cluster','svyid','year','iso3n', 'country', 'region', 'households', 'URBAN_RURA'] + 
                [f'{index1}_bin_{i+1}' for i in range(bin_count)] + 
                [f'{index2}_bin_{i+1}' for i in range(bin_count)]
            ]
    
    if len(index_name.split("_")) == 4:
        index1 = index_name.split("_")[0]
        index2 = index_name.split("_")[1]
        index3 = index_name.split("_")[2]

        features = merged_df[
            	# ['cluster','svyid','year','iso3n', 'country', 'region', 'households', 'URBAN_RURA'] + 
                [f'{index1}_bin_{i+1}' for i in range(bin_count)] + 
                [f'{index2}_bin_{i+1}' for i in range(bin_count)] +
                [f'{index3}_bin_{i+1}' for i in range(bin_count)]
            ]

    target = merged_df['wealthpooled']

    # Drop rows where the target column (wealthpooled) has NaN values
    features = features[~target.isna()]
    target = target.dropna()

    # Convert categorical features to numeric
    features = pd.get_dummies(features)

    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Initialize the regressor
    model = xgb.XGBRegressor(random_state=42)
    param_grid = {}
    if model_name == "xgboost":
        model = xgb.XGBRegressor(random_state=42)
        # Define the parameter grid
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
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        # Define the parameter grid
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20]
        } 

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Make predictions with the best model
    y_pred_best = best_model.predict(X_test)

    # Evaluate the best model
    mse_best = mean_squared_error(y_test, y_pred_best)
    r2_best = r2_score(y_test, y_pred_best)

    # Save the model for future use
    joblib.dump(best_model, f'models/combined_index/wealthpooled/no_extra_features/{model_name}_{index_name}_{bin_count}.pkl')

    result = {
        'Model': model_name,
        'Bins': bin_count,
        'Index': index_name,
        'Best Parameters': best_params,
        'Test R²': r2_best,
        'Test MSE': mse_best
    }

    return result

models = ["xgboost", "gradient_boost", "random_forest"]

df = pd.DataFrame(columns=['Model', 'Bins', 'Index', 'Best Parameters', 'Test R²', 'Test MSE'])

dataset_merged = [
    "../data/processed/merged_datasets/msavi_vari_25_merged.csv",
    "../data/processed/merged_datasets/ndvi_msavi_25_merged.csv",
    "../data/processed/merged_datasets/ndvi_msavi_vari_25_merged.csv",
    "../data/processed/merged_datasets/ndvi_ndbi_25_merged.csv",
    "../data/processed/merged_datasets/ndvi_vari_25_merged.csv",
]

dataset_names = [
    "msavi_vari_25",
    "ndvi_msavi_25",
    "ndvi_msavi_vari_25",
    "ndvi_ndbi_25",
    "ndvi_vari_25",
]

for model in tqdm(models, desc="Loop for each model"):
    count = 0
    for data_src in tqdm(dataset_merged, desc="Loop for each merged dataset"):
        merged_df = pd.read_csv(data_src)
        result = model_train(model, merged_df, dataset_names[count])
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
        count = count + 1

df.to_csv("model_results_combined_wealthpooled_no_extra_features.csv", index=False)
