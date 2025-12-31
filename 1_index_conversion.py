import os
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_ndvi(red, nir):
    np.seterr(divide='ignore', invalid='ignore') 
    red = red / 255.0  
    nir = nir / 255.0  
    ndvi = (nir - red) / (nir + red)
    ndvi = np.nan_to_num(ndvi, nan=-1)
    mean_ndvi = np.nanmean(ndvi)
    return ndvi, mean_ndvi

def calculate_mndwi(green, swir):
    # the formula is same as the one for NDSI index but it depends on the context. NDSI is used to identify snow cover
    np.seterr(divide='ignore', invalid='ignore') 
    green = green / 255.0  
    swir = swir / 255.0  
    mndwi = (green - swir) / (green + swir)
    mndwi = np.nan_to_num(mndwi, nan=-1)
    mean_mndwi = np.nanmean(mndwi)
    return mndwi, mean_mndwi

def calculate_msavi(nir, red):
    np.seterr(divide='ignore', invalid='ignore') 
    nir = nir / 255.0  
    red = red / 255.0  
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
    msavi = np.nan_to_num(msavi, nan=-1)
    mean_msavi = np.nanmean(msavi)
    return msavi, mean_msavi

def calculate_ndbi(swir, nir):
    # (SWIR - NIR) / (SWIR + NIR)
    np.seterr(divide='ignore', invalid='ignore') 
    swir = swir / 255.0  
    nir = nir / 255.0  
    ndbi = (swir - nir) / (swir + nir)
    ndbi = np.nan_to_num(ndbi, nan=-1)
    mean_ndbi = np.nanmean(ndbi)
    return ndbi, mean_ndbi

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
    vari = (green - red) / (green + red - blue)
    vari = np.nan_to_num(vari, nan=-1)
    mean_vari = np.nanmean(vari)
    return vari, mean_vari


def process_images(index):
    # Directories for input and output
    input_dir = '../data/images'
    output_dir = f'../data/processed/{index.upper()}/{index}_images'

    # Create output directories if they do not exist
    os.makedirs(output_dir, exist_ok=True)

    bins_object = {
        "30_bins": [],
        "25_bins": [],
        "20_bins": [],
        "15_bins": [],
        "10_bins": [],
    }

    for bins in tqdm([30, 25, 20, 15, 10], desc=f"{index.upper()} loop for each bin", leave=False):

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

                        if index == "ndvi":
                            # Calculate NDVI
                            index_value, index_mean = calculate_ndvi(red, nir)
                        elif index == "vari":
                            # Calculate VARI
                            index_value, index_mean = calculate_vari(red, green, blue)
                        elif index == "mndwi":
                            # Calculate MNDWI
                            index_value, index_mean = calculate_mndwi(green, swir1)
                        elif index == "msavi":
                            # Calculate MSAVI
                            index_value, index_mean = calculate_msavi(nir, red)
                        elif index == "ndbi":
                            # Calculate NDBI
                            index_value, index_mean = calculate_ndbi(swir1, nir)
                        elif index == "ndmi":
                            # Calculate NDMI
                            index_value, index_mean = calculate_ndmi(nir, swir1)
                        else:
                            continue
                        
                        file_name_without_format = file_name.split(".tif")[0]
                        file_name_parts = file_name_without_format.split("_")

                        LATNUM = float(file_name_parts[0])
                        LONGNUM = float(file_name_parts[1])
                        year = int(file_name_parts[2])

                        # Initialize a dictionary to hold the bin counts for this file
                        bin_counts = {'LATNUM': LATNUM, 'LONGNUM': LONGNUM, 'year': year, f'{index}_mean': index_mean}
                        

                        # Save indexed image
                        index_meta = src.meta
                        index_meta.update(dtype=rasterio.float32, count=1)

                        with rasterio.open(output_tif, 'w', **index_meta) as dst:
                            dst.write(index_value.astype(rasterio.float32), 1)

                        # Generate bin distributions for index
                        bin = np.linspace(-1, 1, bins + 1)
                        hist, bin_edges = np.histogram(index_value, bins=bin)

                        
                        for edge_start, edge_end, count in zip(bin_edges[:-1], bin_edges[1:], hist):
                            bin_label = f"({edge_start}, {edge_end}]"
                            bin_counts[bin_label] = count

                        bins_object[f"{bins}_bins"].append(bin_counts)  

                except Exception as e:
                    print(f'Error processing {file_name}: {e}')

    return bins_object

indices = ["ndvi", "msavi", "ndmi", "ndbi", "mndwi", "vari"]

dhs_data_path = '../data/dhs_wealth_index_cleaned.csv'
dhs_df = pd.read_csv(dhs_data_path)

for index in tqdm(indices, desc="Loop for each index"):
    bins_object = process_images(index)
    output_dir = f'../data/processed/{index.upper()}'

    for key in bins_object.keys():
        bins_df = pd.DataFrame(bins_object[key])
        merged_df = pd.merge(dhs_df, bins_df, on=['LATNUM', 'LONGNUM', 'year'], how="inner")

        output_csv = os.path.join(output_dir, f'{index}_merged_{key}.csv')
        merged_df.to_csv(output_csv, index=False)


print('Index calculation and distribution analysis for all images is complete.')