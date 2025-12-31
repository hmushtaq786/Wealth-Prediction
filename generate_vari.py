import os
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_ndmi(nir, swir1):
    np.seterr(divide='ignore', invalid='ignore') 
    nir = nir / 255.0  
    swir1 = swir1 / 255.0  
    ndmi = (nir - swir1) / (nir + swir1)
    ndmi = np.nan_to_num(ndmi, nan=-1)
    mean_ndmi = np.nanmean(ndmi)
    return ndmi, mean_ndmi

def calculate_vari(red, green, blue):
    # VARI = (Green - Red)/ (Green + Red - Blue)
    np.seterr(divide='ignore', invalid='ignore') 
    red = red / 255.0  
    green = green / 255.0
    blue = blue / 255.0  
    vari = (green - red) / np.where(np.abs(green + red - blue) < 1e-6, np.nan, (green + red - blue))
    vari = np.nan_to_num(vari, nan=-1)
    mean_vari = np.nanmean(vari)
    return vari, mean_vari


def process_images(index):
    # Directories for input and output
    input_dir = '../data/images'
    output_dir = f'../data/processed/{index.upper()}/{index}_images_new1'

    # Create output directories if they do not exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each TIF file
    for file_name in tqdm(os.listdir(input_dir), desc=f"{index.upper()} loop for each file", leave=False):

        if file_name.endswith('.tif'):
            input_tif = os.path.join(input_dir, file_name)
            output_tif = os.path.join(output_dir, f'{index}_{file_name}')
            
            try:
                with rasterio.open(input_tif) as src:

                    if src.count < 4:
                        print(f'File {file_name} does not contain enough bands. Skipping.')
                        continue
                    
                    # Read bands
                    red = src.read(1)
                    green = src.read(2)
                    blue = src.read(3)
                    nir = src.read(4)
                    swir1 = src.read(5)

                    # Check for valid data
                    if red is None or green is None or blue is None or nir is None or swir1 is None:
                        print(f'Error reading bands for {file_name}. Skipping.')
                        continue

                    elif index == "vari":
                        # Calculate VARI
                        index_value, index_mean = calculate_vari(red, green, blue)
                    
                    elif index == "ndmi":
                        # Calculate NDMI
                        index_value, index_mean = calculate_ndmi(nir, swir1)
                    else:
                        continue
                    
                    # Save indexed image
                    index_meta = src.meta
                    index_meta.update(dtype=rasterio.float32, count=1)

                    with rasterio.open(output_tif, 'w', **index_meta) as dst:
                        dst.write(index_value.astype(rasterio.float32), 1)

            except Exception as e:
                print(f'Error processing {file_name}: {e}')

indices = ["vari"]

dhs_data_path = '../data/dhs_wealth_index_cleaned.csv'
dhs_df = pd.read_csv(dhs_data_path)

for index in tqdm(indices, desc="Loop for each index"):
    process_images(index)
    output_dir = f'../data/processed/{index.upper()}'


print('Index calculation and distribution analysis for all images is complete.')