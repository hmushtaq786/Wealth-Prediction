import os
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Directory containing the saved models
models_directory = "models/combined_index/wealth" 
datasets_directory = "../data/processed/merged_datasets"

# List all .pkl files in the directory
model_files = [f for f in os.listdir(models_directory) if f.endswith('.pkl')]
dataset_files = [f for f in os.listdir(datasets_directory) if f.endswith('.csv')]

all_predictions = []
all_true_labels = []

for dataset_file in dataset_files:
    # Load model
    dataset_file_name = dataset_file.split("_merged")[0]  # to remove _merged.csv

    matching_models = [item for item in model_files if "t_"+dataset_file_name in item]  # t_ added to differentiate ndvi_msavi_vari model from msavi_vari model
    
    for model_file in matching_models:
        model_path = os.path.join(models_directory, model_file)
        model = joblib.load(model_path)
        dataset_path = os.path.join(datasets_directory, dataset_file)
        data = pd.read_csv(dataset_path)

        unused_columns = ["wealthpooled", "wealthpooled5country", "wealth", "iso3", "hv000", "cname", "LATNUM", "LONGNUM"]

        if "msavi_mean" in data.columns:
            unused_columns.append("msavi_mean")

        if "vari_mean" in data.columns:
            unused_columns.append("vari_mean")

        if "ndvi_mean" in data.columns:
            unused_columns.append("ndvi_mean")

        if "ndmi_mean" in data.columns:
            unused_columns.append("ndmi_mean")

        if "ndbi_mean" in data.columns:
            unused_columns.append("ndbi_mean")

        if "mndwi_mean" in data.columns:
            unused_columns.append("mndwi_mean")

        features = data.drop(columns=unused_columns)
        print(features.columns)
        features = pd.get_dummies(features)

        imputer = SimpleImputer(strategy='most_frequent')
        features = imputer.fit_transform(features)
        
        target = data["wealth"]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
