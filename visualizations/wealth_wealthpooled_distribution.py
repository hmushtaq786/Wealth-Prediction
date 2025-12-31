import os
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm

dhs_data_path = '../data/dhs_wealth_index_cleaned.csv'
df = pd.read_csv(dhs_data_path)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Plot distributions
plt.figure(figsize=(12, 5))

# Wealth distribution
plt.subplot(1, 2, 1)
sns.histplot(df["wealth"], bins=50, kde=True, color="blue")
plt.title("Distribution of Wealth")
plt.xlabel("Wealth Index")
plt.ylabel("Density")

# Wealthpooled distribution
plt.subplot(1, 2, 2)
sns.histplot(df["wealthpooled"], bins=50, kde=True, color="green")
plt.title("Distribution of Wealthpooled")
plt.xlabel("Wealthpooled Index")
plt.ylabel("Density")

plt.tight_layout()
plt.show()
