import os
import pandas as pd

# Define the path to the folder containing the text files and the CSV file
# folder_path = '../data/ndvi_bins'

csv_file_path = '../data/dhs_wealth_index_cleaned.csv'

dataset_combinations = {
    "msavi_vari": ["../data/processed/MSAVI/msavi_merged_25_bins.csv", "../data/processed/VARI/vari_merged_25_bins.csv"],
    "ndvi_msavi_vari": ["../data/processed/NDVI/ndvi_merged_25_bins.csv", "../data/processed/MSAVI/msavi_merged_25_bins.csv", "../data/processed/VARI/vari_merged_25_bins.csv"],
    "ndvi_msavi": ["../data/processed/NDVI/ndvi_merged_25_bins.csv", "../data/processed/MSAVI/msavi_merged_25_bins.csv"],
    "ndvi_ndbi": ["../data/processed/NDVI/ndvi_merged_25_bins.csv", "../data/processed/NDBI/ndbi_merged_25_bins.csv"],
    "ndvi_vari": ["../data/processed/NDVI/ndvi_merged_25_bins.csv", "../data/processed/VARI/vari_merged_25_bins.csv"],
}

key_columns = ['LATNUM', 'LONGNUM', 'year']

for key in dataset_combinations.keys():
    combination = dataset_combinations[key]

    index1 = pd.read_csv(combination[0])
    index2 = pd.read_csv(combination[1])

    index1_last_25_columns = index1.iloc[:, -25:]
    index2_last_25_columns = index2.iloc[:, -25:]

    index1_new_column_names = {col: f"{key.split('_')[0]}_bin_{i+1}" for i, col in enumerate(index1_last_25_columns)}  # Generate new names
    index2_new_column_names = {col: f"{key.split('_')[1]}_bin_{i+1}" for i, col in enumerate(index2_last_25_columns)}  # Generate new names

    # Apply the renaming
    index1.rename(columns=index1_new_column_names, inplace=True)
    index2.rename(columns=index2_new_column_names, inplace=True)

    index1_last_26_columns = index1.iloc[:, -26:]
    index2_last_26_columns = index2.iloc[:, -26:]

    # Add the merge columns ('LATNUM', 'LONGNUM', 'year') to the extracted columns
    df2_to_merge = pd.concat([index1[['LATNUM', 'LONGNUM', 'year']], index2_last_26_columns], axis=1)
    
    merged_df = pd.merge(index1, df2_to_merge, on=['LATNUM', 'LONGNUM', 'year'], how='outer')

    if len(combination) == 3:
        index3 = pd.read_csv(combination[2])
        index3_last_25_columns = index3.iloc[:, -25:]
        index3_new_column_names = {col: f"{key.split('_')[2]}_bin_{i+1}" for i, col in enumerate(index3_last_25_columns)}  # Generate new names
        index3.rename(columns=index3_new_column_names, inplace=True)
        index3_last_26_columns = index3.iloc[:, -26:]
        df3_to_merge = pd.concat([index1[['LATNUM', 'LONGNUM', 'year']], index3_last_26_columns], axis=1)
        merged_df = pd.merge(merged_df, df3_to_merge, on=['LATNUM', 'LONGNUM', 'year'], how='outer')

    output_file_path = os.path.join(f"../data/processed/merged_datasets/{key}_25_merged.csv")

    merged_df.to_csv(output_file_path, index=False)
