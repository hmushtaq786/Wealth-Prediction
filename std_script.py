import os
import numpy as np
import rasterio
from tqdm import tqdm

# Choose index: "vari" or "ndmi"
index = "vari"

# Path to folder containing your .tif files
DATA_DIR = f'../data/processed/{index.upper()}/{index}_images_new1'

def get_valid_pixels(filepath):
    with rasterio.open(filepath) as src:
        arr = src.read().astype(np.float32)
        nodata = src.nodata

        # Replace nodata with NaN
        if nodata is not None:
            arr[arr == nodata] = np.nan

        # Flatten and remove NaN / Inf
        arr = arr[np.isfinite(arr)]

        # Skip empty tiles
        if arr.size == 0:
            # print(f"⚠️  Skipping {os.path.basename(filepath)} (no valid data)")
            return None

        max_val = np.nanmax(arr)
        if max_val > 1000:
            arr = arr / 10000.0
        elif max_val > 10:
            print(f"⚠️  Suspiciously large values in {os.path.basename(filepath)} (max={max_val}); check scaling.")

        return arr


def compute_stats(keyword):
    all_pixels = []

    files = [f for f in os.listdir(DATA_DIR)
             if f.lower().endswith(".tif") and keyword in f.lower()]

    if not files:
        print(f"⚠️  No .tif files found for {keyword.upper()} in {DATA_DIR}")
        return

    for fname in tqdm(files, desc=f"{keyword.upper()} loop for each file", leave=False):
        fpath = os.path.join(DATA_DIR, fname)
        arr = get_valid_pixels(fpath)
        if arr is not None and arr.size > 0:
            all_pixels.append(arr)

    if not all_pixels:
        print(f"⚠️  No valid pixel data found for {keyword.upper()}.")
        return

    # Concatenate and compute global stats
    all_pixels = np.concatenate(all_pixels)
    mean_val = np.mean(all_pixels)
    std_val = np.std(all_pixels)

    print(f"\n=== {keyword.upper()} Dataset Statistics ===")
    print(f"Global Mean: {mean_val:.6f}")
    print(f"Global Std:  {std_val:.6f}\n")


if __name__ == "__main__":
    compute_stats(index)
