import os
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
import seaborn as sns

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT_IMAGE_FOLDER = os.environ.get('WS_DIR') + '/processed'
INDEX_NAMES = ["NDVI", "MSAVI", "VARI", "NDBI", "NDMI", "MNDWI"]
TARGET_SIZE = (224, 224)   # Resize everything to 224x224

# ---------------------------------------------------------------------
# LOAD AND RESIZE .TIF IMAGES
# ---------------------------------------------------------------------
def load_and_resize_tif_stack(folder):
    folder = Path(folder)
    tif_files = sorted(list(folder.glob("*.tif")))

    if len(tif_files) == 0:
        raise ValueError(f"No .tif files found in: {folder}")

    resized_images = []

    for tif in tif_files:
        with rasterio.open(tif) as src:
            arr = src.read(1).astype(np.float32)

            # --- Resize to 224x224 ---
            arr_resized = resize(
                arr,
                TARGET_SIZE,
                mode='reflect',
                preserve_range=True,
                anti_aliasing=True
            )

            resized_images.append(arr_resized)

    # Average if multiple tif files exist
    if len(resized_images) > 1:
        return np.mean(np.stack(resized_images, axis=0), axis=0)

    return resized_images[0]

# ---------------------------------------------------------------------
# LOAD ALL INDEX ARRAYS
# ---------------------------------------------------------------------
data_arrays = {}

for idx in INDEX_NAMES:
    print(f"Loading index: {idx}")
    folder = os.path.join(ROOT_IMAGE_FOLDER, idx, f"{idx.lower()}_images")
    data_arrays[idx] = load_and_resize_tif_stack(folder)

# ---------------------------------------------------------------------
# COMBINE + FLATTEN
# ---------------------------------------------------------------------
flat = {name: data_arrays[name].ravel() for name in INDEX_NAMES}
matrix = np.vstack([flat[name] for name in INDEX_NAMES]).T
matrix = matrix[~np.isnan(matrix).any(axis=1)]
matrix = matrix[~np.isinf(matrix).any(axis=1)]

# ---------------------------------------------------------------------
# SPEARMAN CORRELATION
# ---------------------------------------------------------------------
corr, pvals = spearmanr(matrix, axis=0)
corr_matrix = corr[:len(INDEX_NAMES), :len(INDEX_NAMES)]

corr_df = pd.DataFrame(corr_matrix, index=INDEX_NAMES, columns=INDEX_NAMES)
print("\nSpearman Correlation Matrix:\n")
print(corr_df)

OUT_DIR = "optuna/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# CORRELATION HEATMAP ONLY
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm")
plt.colorbar(label="Spearman Correlation")

plt.xticks(range(len(INDEX_NAMES)), INDEX_NAMES, rotation=45, ha='right')
plt.yticks(range(len(INDEX_NAMES)), INDEX_NAMES)
plt.title("Spearman Correlation Matrix")
plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap_latest1.png"), dpi=300)
plt.close()

# ---------------------------------------------------------------------
# SCATTERPLOT MATRIX ONLY
# ---------------------------------------------------------------------
num_vars = len(INDEX_NAMES)
fig, axes = plt.subplots(num_vars, num_vars, figsize=(16, 16))

for i in range(num_vars):
    for j in range(num_vars):
        ax = axes[i, j]

        if i == j:
            ax.hist(matrix[:, i], bins=40, color="gray")
        else:
            ax.scatter(matrix[:, j], matrix[:, i], s=1, alpha=0.3)

        # Label edges only
        if i == num_vars - 1:
            ax.set_xlabel(INDEX_NAMES[j])
        else:
            ax.set_xticks([])

        if j == 0:
            ax.set_ylabel(INDEX_NAMES[i])
        else:
            ax.set_yticks([])

plt.suptitle("Scatterplot Matrix", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig(os.path.join(OUT_DIR, "scatterplot_matrix_latest1.png"), dpi=300)
plt.close()

# Convert your matrix to a DataFrame
df = pd.DataFrame(matrix, columns=INDEX_NAMES)

# Seaborn pairplot
g = sns.pairplot(
    df,
    diag_kind="hist",      # histogram on diagonal
    plot_kws={"s": 5, "alpha": 0.3},  # point size + transparency
    corner=False           # show full matrix
)

# Rotation fix
for ax in g.axes.flatten():
    if ax is not None:
        ax.tick_params(axis='x', labelrotation=45)
        ax.ticklabel_format(style='plain', axis='both')

plt.suptitle("Scatterplot Matrix (Seaborn)", fontsize=18, y=1.02)
plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "scatterplot_matrix_sns.png"), dpi=300)
plt.close()

# ---------------------------------------------------------------------
# HISTOGRAMS ONLY (EACH INDEX)
# ---------------------------------------------------------------------
num_vars = len(INDEX_NAMES)
rows = int(np.ceil(num_vars / 3))
cols = 3

fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
axes = axes.flatten()

for i, name in enumerate(INDEX_NAMES):
    ax = axes[i]
    ax.hist(matrix[:, i], bins=40, color="steelblue")
    ax.locator_params(axis='x', nbins=5)
    ax.set_title(f"Histogram of {name}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

# Hide unused subplots
for k in range(len(INDEX_NAMES), len(axes)):
    axes[k].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "histograms_latest1.png"), dpi=300)
plt.close()

