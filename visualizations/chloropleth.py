import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib
from sklearn.impute import SimpleImputer
import numpy as np

def plot_africa_with_data_collections_location_count_and_wealth_data(original_df):
    # Load manually downloaded shapefile
    world = gpd.read_file("visualizations/map_shape/ne_110m_admin_0_countries.shp")  # Change to actual path
    africa = world[world['CONTINENT'] == 'Africa']  # Column names may differ

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    vmin = min(original_df['predicted'].min(), original_df['actual'].min())
    vmax = max(original_df['predicted'].max(), original_df['actual'].max())

    # First plot: Data collection count
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    africa.plot(ax=ax[0], color='lightgray', edgecolor='black')
    scatter1 = ax[0].scatter(original_df['LONGNUM'], original_df['LATNUM'], 
                             c=original_df['actual'], cmap='viridis', 
                             s=10, alpha=0.6, vmin=vmin, vmax=vmax
                             )
    fig.colorbar(scatter1, ax=ax[0], cax=cax, label='Wealth Index')
    ax[0].set_title("Actual Wealthpooled Index at Data Collection Sites", fontsize=12)

    # Second plot: Wealth Index
    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    africa.plot(ax=ax[1], color='lightgray', edgecolor='black')
    scatter2 = ax[1].scatter(original_df['LONGNUM'], original_df['LATNUM'], 
                             c=original_df['predicted'], cmap='viridis', 
                             s=10, alpha=0.6, vmin=vmin, vmax=vmax
                             )
    fig.colorbar(scatter2, ax=ax[1], cax=cax2, label='Wealth Index')
    ax[1].set_title("Predicted Wealthpooled Index at Data Collection Sites", fontsize=12)

    plt.tight_layout(pad=2.5)
    plt.show()

# Load dataset
pred_data = pd.read_csv("preds.csv")

# Plot maps
plot_africa_with_data_collections_location_count_and_wealth_data(pred_data)
