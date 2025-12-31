import pandas as pd
import joblib
import numpy as np
from sklearn.impute import SimpleImputer

# === Step 1: Load Trained Model ===
model = joblib.load("models/combined_index/wealthpooled/xgboost_msavi_vari_25_25.pkl")

# === Step 2: Load and Prepare Dataset ===
dataset_path = "../data/processed/merged_datasets/msavi_vari_25_merged.csv" 
df = pd.read_csv(dataset_path)

# If necessary, preprocess (e.g., scaling, encoding)
# X = preprocess(X)  # Uncomment if needed
bin_count = 25

index1 = "msavi"
index2 = "vari"

features = df[
            ['LATNUM', 'LONGNUM', 'wealthpooled', 'cluster','svyid','year','iso3n', 'country', 'region', 'households', 'URBAN_RURA'] + 
            [f'{index1}_bin_{i+1}' for i in range(bin_count)] + 
            [f'{index2}_bin_{i+1}' for i in range(bin_count)]
        ]

target = df['wealthpooled']

# Drop rows where the target column (wealth) has NaN values
features = features[~target.isna()]
target = target.dropna()

# Convert categorical features to numeric
features = pd.get_dummies(features)

imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(features), columns=features.columns) 

# === Step 4: Predict One-by-One and Store Results ===

preds = {
    "actual": [],
    "LONGNUM": [],
    "LATNUM": [],
    "predicted": [],
    "URBAN_RURA": []
}

for index, row in X.iterrows():
    if row.URBAN_RURA_U == 1.0:
        preds["URBAN_RURA"].append("U")
    else:
        preds["URBAN_RURA"].append("R")

    actual = row["wealthpooled"]

    preds["actual"].append(actual)
    preds["LONGNUM"].append(row["LONGNUM"])
    preds["LATNUM"].append(row["LATNUM"])

    row = row.drop(labels=["wealthpooled", "LONGNUM", "LATNUM"])
    single_record = np.array(row).reshape(1, -1)  # Convert to model input shape
    predicted_value = model.predict(single_record)[0]  # Get prediction
    # preds["predicted"] = predicted_value

    preds["predicted"].append(predicted_value)
    print(f"Record {index}: Predicted Value = {predicted_value}", f"Actual Value = {actual}")

# === Step 5: Save Predictions to CSV ===
df_p = pd.DataFrame(preds)
df_p.to_csv("predictions_to_visualize.csv", index=False)
